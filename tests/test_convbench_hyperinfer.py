"""CPU contracts of the ConvBench HyperInfer run (docs/convbench_hyperinfer_contract.md;
scripts 231 / 232 / 233 / 234 / 235, scripts/convbench_hyperinfer_common.py).

No model is loaded.  What is pinned:

* the run's arm and phase tables live in a private copy of ``convbench_main_common``; the module other code
  imports keeps the tables of the main experiment;
* every arm maps to an arm of the imported harness (scripts/209, scripts/206) with the IMPRESS store format, and
  the HyperInfer request is the request function of the MT-GQA run (scripts/228, scripts/225), not a copy;
* nothing an earlier run froze has changed, and the HyperInfer module is the one the MT-GQA run measured;
* the gate comparisons (output identity, prefetch off, unread rows, repeat, stop check, recorded IMPRESS tokens),
  the admission and identity rules, the driver plan and the report statistics on synthetic rows.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_main_common as SHARED  # noqa: E402
import convbench_hyperinfer_common as HC  # noqa: E402

C = HC.C
MAIN_CFG = json.loads((HC.MAIN_RUN / "config.json").read_text())
MT_CFG = json.loads((HC.MTGQA_RUN / "config.json").read_text())


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def same_environment():
    import numpy
    import torch
    import transformers
    env = MAIN_CFG["environment"]
    return (transformers.__version__, torch.__version__, numpy.__version__, sys.executable) == (env["transformers"], env["torch"], env["numpy"], env["executable"])


def _hyper(late=0, after=1, layers=4, prefetched=1000):
    cols = {"late_starts": [0] + [late] + [0] * (layers - 2), "reads_done_after_stop": [0] + [after] * (layers - 1),
            "k_done": [0] + [2] * (layers - 1), "v_done": [0] + [1] * (layers - 1), "begin_after_stop_ms": [None] + [-0.5] * (layers - 1)}
    return {"truncation": "reactive", "layers": cols, "late_starts": late, "jobs": layers - 1, "prefetched_bytes": prefetched}


def _row(role, turn, tokens, status="ok", digest="aa", **extra):
    hit = turn > 1
    row = {"role": role, "method": HC.DISPLAY[role], "turn_id": turn, "status": status,
           "generated_token_ids": list(tokens) if status == "ok" else None, "generated_tokens": len(tokens) if status == "ok" else None,
           "ttft_ms": 100.0 + turn, "request_e2e_ms": 200.0, "prefill_ms": 90.0, "hook_ms": 30.0, "read_blocked_ms": 10.0 if hit else None,
           "ssd_read_bytes": 1000 if hit else 0, "probe_read_bytes": 100 if hit else 0, "k_read_bytes": 450 if hit else 0, "v_read_bytes": 450 if hit else 0,
           "read_calls": 5 if hit else 0, "pread_calls": 5 if hit else 0, "preadv_calls": 0, "input_tokens": 1000 + 100 * turn,
           "input_above_4096": False, "cap_reached": False, "result": {}}
    if hit and role != HC.FULL:
        row["result"]["selection_digest"] = {"all_layers": digest, "per_layer": [digest] * 4}
    if hit and role in HC.PREFETCH_ROLES:
        row["result"]["hyperinfer"] = _hyper()
    if hit and role == HC.HYPER_NAN:
        row["result"]["unread_row_guard"] = {"all_outputs_finite": True, "used_equals_selection": True, "used_rows_all_read": True,
                                             "unread_rows": 7, "used_rows": 3, "attention_calls": 4}
    row.update(extra)
    return row


def _doc(cid, roles, tokens_of=lambda role, t: [1, 2, t]):
    return {"conversation_id": cid, "rows": [_row(role, t, tokens_of(role, t)) for t in (1, 2, 3) for role in roles], "failures": [], "stores": {},
            "roles": roles, "seconds": 10.0}


class Tables(unittest.TestCase):
    def test_shared_module_keeps_the_main_experiment_tables(self):
        self.assertIsNot(C, SHARED)
        self.assertEqual(SHARED.MAIN_ROLES, ["recomp", "fullload", "impress", "ours"])
        self.assertEqual(len(SHARED.GATE_ROLES), 9)
        self.assertEqual(SHARED.RUN_PREFIX, "convbench_main_")
        self.assertEqual(SHARED.CONTRACT.name, "convbench_main_contract.md")
        self.assertIs(sys.modules["convbench_main_common"], SHARED)

    def test_this_run_tables(self):
        self.assertEqual(C.MAIN_ROLES, [HC.HYPER])
        self.assertEqual([HC.DISPLAY[r] for r in HC.FIVE], ["ReComp", "FullLoad", "IMPRESS", "HyperInfer", "Ours"])
        self.assertEqual([HC.DISPLAY[r] for r in HC.PILOT_ROLES], ["FullLoad", "IMPRESS", "HyperInfer"])
        self.assertEqual(len(HC.GATE_ROLES), 6)
        self.assertEqual(C.PHASE_ROLES, {"gate": HC.GATE_ROLES, "pilot": HC.PILOT_ROLES, "overflow": [HC.HYPER], "main": [HC.HYPER]})
        self.assertEqual(C.RUN_PREFIX, "convbench_hyperinfer_")
        self.assertEqual(C.CONTRACT.name, "convbench_hyperinfer_contract.md")
        self.assertTrue(C.CONTRACT.is_file())
        # the conversations of the gate and the pilot are those of the main experiment
        self.assertEqual((C.GATE_ORDINALS, C.PILOT_ORDINALS), (SHARED.GATE_ORDINALS, SHARED.PILOT_ORDINALS))
        self.assertEqual((C.N_CONV, C.MAX_NEW, C.SEED, C.RESAMPLES), (577, 1024, 1234, 10_000))

    def test_loaded_scripts_bind_to_the_private_copy(self):
        E = HC.load_with_common("_convbench_hyperinfer_runner", "217_eval_convbench_main.py")
        self.assertIs(E.C, C)
        self.assertIs(sys.modules["convbench_main_common"], SHARED)
        other = _load("_test_convbench_hyperinfer_plain_217", "217_eval_convbench_main.py")
        self.assertIs(other.C, SHARED)

    def test_rotation_is_balanced(self):
        index, _ = C.load_index()
        for phase, n in (("pilot", 40), ("gate", 4)):
            roles = C.PHASE_ROLES[phase]
            first = [C.rotate(roles, index[o]["ordinal"])[0] for o in C.phase_ordinals(None, "next", phase)]
            self.assertEqual(len(first), n)
            if phase == "pilot":
                counts = [first.count(r) for r in roles]
                self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(C.rotate(HC.MAIN_ROLES, 5), [HC.HYPER])

    def test_recorded_rows_of_the_main_experiment(self):
        index, _ = C.load_index()
        cid = index[C.PILOT_ORDINALS[0]]["conversation_id"]
        rows = HC.recorded_rows("next", cid)
        self.assertEqual({r for r, _ in rows}, {"recomp", "fullload", "impress", "ours"})
        self.assertTrue(all(HC.tokens(rows[(C.IMP, t)]) for t in (1, 2, 3)))
        self.assertEqual(HC.impress_kind(HC.recorded_doc("next", cid)), "impress_official")
        self.assertIsNone(HC.recorded_rows("next", "convbench:does-not-exist"))


class Harness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.EV = _load("_test_convbench_hyperinfer_eval", "231_eval_convbench_hyperinfer.py")
        cls.backends = {m: cls.EV.Backend(m) for m in C.MODELS}

    @classmethod
    def tearDownClass(cls):
        for be in cls.backends.values():      # the harness modules are shared with other tests of this process
            for arm in HC.BACKEND_ARM.values():
                be.B.ARMS.pop(arm, None)

    def test_arms_map_to_the_harness(self):
        for model, be in self.backends.items():
            B = be.B
            self.assertEqual(set(be.arms), set(HC.GATE_ROLES))
            self.assertEqual((be.family(HC.FULL), be.kind(HC.FULL)), ("fullload", "raster"))
            self.assertEqual(be.family(HC.IMP), "impress_official")
            self.assertTrue(B.ARMS[be.arms[HC.IMP]]["async"] and B.ARMS[be.arms[HC.IMP]]["ratio"] == 0.25)
            for role, arm in HC.BACKEND_ARM.items():
                self.assertEqual(be.arms[role], arm)
                self.assertEqual(be.family(role), "hyperinfer")
                self.assertEqual(be.kind(role), be.kind(HC.IMP), "HyperInfer reads the IMPRESS store format")
                self.assertEqual(dict(B.ARMS[arm]), MT_CFG["models"][HC.MTGQA_MODEL[model]]["arms"][arm], "the arm entry of the MT-GQA run")
            self.assertEqual(MT_CFG["models"][HC.MTGQA_MODEL[model]]["impress_official"], B.OFFICIAL)

    def test_prefetch_switches(self):
        arms = self.EV.M228.HYPER_ARMS
        self.assertEqual(arms[HC.BACKEND_ARM[HC.HYPER]], {"kv_prefetch": "async", "nan": False, "display": "HyperInfer"})
        self.assertEqual(arms[HC.BACKEND_ARM[HC.HYPER_OFF]]["kv_prefetch"], "off")
        self.assertEqual((arms[HC.BACKEND_ARM[HC.HYPER_NAN]]["kv_prefetch"], arms[HC.BACKEND_ARM[HC.HYPER_NAN]]["nan"]), ("async", True))
        self.assertEqual(arms[HC.BACKEND_ARM[HC.HYPER_REPEAT]]["kv_prefetch"], "async")
        self.assertEqual(MT_CFG["hyperinfer_arms"], {k: dict(v) for k, v in arms.items()})

    def test_request_functions_are_those_of_the_mtgqa_run(self):
        EV = self.EV
        for be in self.backends.values():
            req = be.requests
            self.assertIs(type(req).hyper_hit, EV.M228.Backend.hyper_hit)
            self.assertIs(type(req).hyper_request, EV.M228.E1.Backend.hyper_request)
            self.assertIs(type(req).dispatch, EV.M228.E1.Backend.dispatch)
            self.assertIs(req.SRC, be.B)
            self.assertIs(EV.M228.H2, EV.H2)
        self.assertEqual(C.sha(ROOT / "mmimpress/hyperinfer_v2.py"), MT_CFG["hyperinfer"]["module_sha256"])

    def test_stored_request_goes_through_the_dispatcher(self):
        be = self.backends["llava15"]
        seen = []
        with mock.patch.object(be.requests, "hyper_request", lambda *a: seen.append(("hyper", a[4])) or ("r", "d")), \
                mock.patch.object(be.requests, "_stored_request", lambda *a, **k: seen.append(("stored", a[4])) or ({}, "d")):
            be.runner, be.servers = "runner", {"base": None}
            self.assertEqual(be.stored("ctx", "factory", HC.HYPER, "img", "path"), ("r", "d"))
            be.stored("ctx", "factory", HC.HYPER_OFF, "img", "path")
            be.stored("ctx", "factory", HC.FULL, "img", "path")
        self.assertEqual(seen, [("hyper", "hyperinfer_async"), ("hyper", "hyperinfer_off"), ("stored", "fullload_async")])

    def test_row_additions(self):
        EV = self.EV
        be = SimpleNamespace(family=lambda role: "hyperinfer", arms={HC.HYPER: "hyperinfer_async"}, B=SimpleNamespace(safe=lambda x: dict(x)),
                             runner=SimpleNamespace(processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2))))
        result = {"generated_token_ids": [5, 6, 2], "answer": "a b", "first_token_id": 5, "vision_forward_count": 0, "end_to_end_ttft_ms": 12.0,
                  "request_e2e_ms": 20.0, "prefill_ms": 9.0, "hook_ms": 4.0, "io": {"bytes": 64, "per_kind": {"k": {"bytes": 32}, "v": {"bytes": 32}}},
                  "pread_calls": 2, "preadv_calls": 0, "read_wait": {"read_blocked_ms": 1.5, "readahead": None}, "cap_reached": False,
                  "layer_log": [{"layer": 0, "mode": "consensus", "sim": 0.5, "k_chunks": 2, "v_chunks": 2}, {"layer": 1, "mode": "fallback", "sim": 0.1, "k_chunks": 9, "v_chunks": 9}],
                  "hyperinfer": {"truncation": "reactive"}, "selection_digest": {"all_layers": "x", "per_layer": ["a", "b"]}}
        geometry = {"prefix_len": 611, "v_token_num": 576}
        row = EV.make_row(be, "main", HC.HYPER, 2, [HC.HYPER], "prompt", result, 700, True, geometry)
        self.assertEqual(row["suffix_tokens"], 89)
        self.assertEqual(row["impress"], {"fb": [0, 1], "sim": [0.5, 0.1], "kc": [2, 9], "vc": [2, 9], "nc": 9})
        self.assertEqual(row["method"], "HyperInfer")
        self.assertEqual(row["generated_token_ids"], [5, 6, 2])
        self.assertNotIn("layer_log", row["result"])                         # dropped outside the gate phase ...
        self.assertEqual(row["result"]["hyperinfer"], {"truncation": "reactive"})      # ... the prefetch record is kept
        self.assertEqual(row["result"]["selection_digest"]["all_layers"], "x")
        self.assertIs(row["result"]["cap_reached"], False)
        self.assertEqual(row["read_blocked_ms"], 1.5)

    def test_phase_estimate(self):
        costs = {"a": {"write_impress": 10, "write_fullload": 4, "e2e_impress_s": 2.0, "e2e_fullload_s": 1.0, "store_impress_s": 0.5, "store_fullload_s": 0.25,
                       "other_s_per_request": 0.1}}
        self.assertEqual(self.EV.phase_estimate(costs, ["a"], HC.MAIN_ROLES), (10, 2.0 + 0.5 + 0.3))
        w, s = self.EV.phase_estimate(costs, ["a"], HC.PILOT_ROLES)
        self.assertEqual(w, 14)
        self.assertAlmostEqual(s, 2 * 2.0 + 0.5 + 0.9 + 1.0 + 0.25)

    def test_earlier_runs_frozen_files_are_unchanged(self):
        self.assertEqual(C.freeze_violations(MAIN_CFG["source_sha256"]), [])
        self.assertEqual(C.freeze_violations({k: v for k, v in MT_CFG["source_sha256"].items() if not C.REPORT_SCRIPT.match(k)}), [])
        self.assertEqual(C.sha(HC.PORT_TABLE), MT_CFG["port_table_sha256"])
        self.assertEqual(C.sha(HC.MAIN_CONTRACT), MAIN_CFG["contract_sha256"])

    @unittest.skipUnless(same_environment(), "the freeze checks compare with the environment of the earlier runs")
    def test_freeze_checks_pass(self):
        models, changed, _main, _mt, now = self.EV.freeze_checks()
        self.assertEqual(changed, {"convbench_main": [], "mt_gqa_hyperinfer_v2": []})
        self.assertEqual(set(models), set(C.MODELS))
        for m in C.MODELS:
            self.assertEqual(set(models[m]["hyperinfer_arms"]), set(HC.BACKEND_ARM))
        self.assertIn("scripts/231_eval_convbench_hyperinfer.py", now)
        self.assertNotIn("scripts/233_report_convbench_hyperinfer.py", now)      # the report script is not frozen


class Gates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V = _load("_test_convbench_hyperinfer_gates", "232_validate_convbench_hyperinfer.py")

    def docs(self, roles=HC.GATE_ROLES, n=2):
        return {f"c{i}": _doc(f"c{i}", roles) for i in range(n)}

    def test_output_identity(self):
        V = self.V
        docs = self.docs()
        out = V.compare(docs, HC.HYPER, HC.IMP, selection=True)
        self.assertTrue(out["ok"])
        self.assertEqual((out["requests"], out["followup_requests"]), (6, 4))
        C.rows_by_key(docs["c1"])[(HC.HYPER, 3)]["generated_token_ids"][-1] = 99
        out = V.compare(docs, HC.HYPER, HC.IMP, selection=True)
        self.assertFalse(out["ok"])
        self.assertEqual(out["mismatches"], 1)
        self.assertIn("first difference at 2", out["first_mismatches"][0]["difference"][0])

    def test_selection_and_bytes_and_calls(self):
        V = self.V
        docs = self.docs()
        C.rows_by_key(docs["c0"])[(HC.HYPER, 2)]["result"]["selection_digest"] = {"all_layers": "bb", "per_layer": ["aa", "bb", "aa", "aa"]}
        out = V.compare(docs, HC.HYPER, HC.IMP, selection=True)
        self.assertEqual(out["first_mismatches"][0]["difference"], ["selection (layers [1])"])
        self.assertTrue(V.compare(docs, HC.HYPER, HC.IMP)["ok"])      # tokens alone are equal
        off = C.rows_by_key(docs["c1"])[(HC.HYPER_OFF, 2)]
        off["k_read_bytes"] += 64; off["ssd_read_bytes"] += 64; off["pread_calls"] += 1; off["read_calls"] += 1
        out = V.compare(docs, HC.HYPER_OFF, HC.IMP, fields=V.BYTE_FIELDS + V.CALL_FIELDS)
        self.assertFalse(out["ok"])
        self.assertEqual(len(out["first_mismatches"][0]["difference"]), 4)

    def test_failed_request_is_a_mismatch(self):
        V = self.V
        docs = self.docs()
        r = C.rows_by_key(docs["c0"])[(HC.HYPER, 2)]
        r["status"], r["generated_token_ids"] = "failed", None
        self.assertFalse(V.compare(docs, HC.HYPER, HC.IMP)["ok"])
        self.assertFalse(V.stop_check(docs, [HC.HYPER])["ok"])
        self.assertFalse(V.compare({}, HC.HYPER, HC.IMP)["ok"])        # nothing compared is not a pass

    def test_stop_check(self):
        V = self.V
        docs = self.docs()
        out = V.stop_check(docs, HC.PREFETCH_ROLES)
        self.assertTrue(out["ok"])
        self.assertEqual(out["arms"][HC.HYPER]["layers_checked"], 4 * 3)
        self.assertEqual(out["arms"][HC.HYPER]["reads_started_after_stop"], 0)
        C.rows_by_key(docs["c0"])[(HC.HYPER_REPEAT, 2)]["result"]["hyperinfer"] = _hyper(late=1)
        out = V.stop_check(docs, HC.PREFETCH_ROLES)
        self.assertFalse(out["ok"])
        self.assertEqual(out["first_violations"][0]["layers_with_late_starts"], [1])
        docs = self.docs()
        C.rows_by_key(docs["c0"])[(HC.HYPER, 3)]["result"]["hyperinfer"] = _hyper(after=2)      # two reads ended after the stop
        self.assertFalse(V.stop_check(docs, [HC.HYPER])["ok"])

    def test_unread_row_guard(self):
        V = self.V
        docs = self.docs()
        out = V.nan_guard(docs)
        self.assertTrue(out["ok"])
        self.assertEqual(out["totals"], {"unread_rows": 28, "used_rows": 12, "attention_calls": 16})
        C.rows_by_key(docs["c1"])[(HC.HYPER_NAN, 3)]["result"]["unread_row_guard"]["used_rows_all_read"] = False
        self.assertFalse(V.nan_guard(docs)["ok"])
        docs = self.docs()
        del C.rows_by_key(docs["c1"])[(HC.HYPER_NAN, 3)]["result"]["unread_row_guard"]
        self.assertFalse(V.nan_guard(docs)["ok"])

    def test_against_recorded_impress(self):
        V = self.V
        docs = self.docs(HC.MAIN_ROLES, n=3)
        ref = {cid: {(C.IMP, t): _row(C.IMP, t, [1, 2, t]) for t in (1, 2, 3)} for cid in docs}
        ref["c1"][(C.IMP, 2)]["generated_token_ids"] = [1, 7, 2, 8]
        ref["c1"][(C.IMP, 3)]["generated_token_ids"] = [4]
        with mock.patch.object(HC, "recorded_rows", lambda model, cid: ref.get(cid)):
            out = V.against_recorded("next", docs)
        self.assertFalse(out["ok"])
        self.assertEqual((out["requests"], out["equal"], out["mismatches"]), (9, 7, 2))
        self.assertEqual(out["equal_per_turn"], {"T1": [3, 3], "T2": [2, 3], "T3": [2, 3]})
        first, second = out["mismatch_list"]
        self.assertEqual((first["turn_id"], first["earlier_turns_of_this_conversation_equal"], first["first_differing_position"]), (2, True, 1))
        self.assertEqual((second["turn_id"], second["earlier_turns_of_this_conversation_equal"]), (3, False))
        ref["c1"] = {(C.IMP, t): _row(C.IMP, t, [1, 2, t]) for t in (1, 2, 3)}
        with mock.patch.object(HC, "recorded_rows", lambda model, cid: ref.get(cid)):
            self.assertTrue(V.against_recorded("next", docs)["ok"])

    def test_turn1_identical_and_ratio(self):
        V = self.V
        docs = self.docs(HC.PILOT_ROLES)
        self.assertEqual(V.t1_identical(docs, HC.PILOT_ROLES)["identical"], 2)
        C.rows_by_key(docs["c0"])[(HC.FULL, 1)]["generated_token_ids"] = [9]
        self.assertEqual(V.t1_identical(docs, HC.PILOT_ROLES)["identical"], 1)
        a = np.array([200.0, 220.0, 240.0]); b = np.array([400.0, 440.0, 480.0])
        r = V.ratio_ci(a, b)
        self.assertAlmostEqual(r["ratio"], 0.5)
        self.assertAlmostEqual(r["ci_low"], 0.5); self.assertAlmostEqual(r["ci_high"], 0.5)
        self.assertAlmostEqual(r["diff_ms"], -110.0)
        self.assertEqual(V.ratio_ci(a, b), V.ratio_ci(a, b))      # seeded

    def test_gate_result_is_written_once(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.V.gate_path(tmp, "next", "gate_gate")
            self.V.write_once(path, {"status": "FAIL"}, {"_file_sha256": "x"})
            with self.assertRaises(FileExistsError):
                self.V.write_once(path, {"status": "PASS"}, {"_file_sha256": "x"})
            self.assertEqual(C.read_json(path)["status"], "FAIL")


class DriverAndJudge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = _load("_test_convbench_hyperinfer_driver", "234_convbench_hyperinfer_driver.py")

    def test_plan_order(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = self.D.Driver(Path(tmp) / "convbench_hyperinfer_x")
            self.assertEqual(d.plan(), ["tests", "gate:next", "pilot:next", "main:next", "gate:llava15", "pilot:llava15", "main:llava15",
                                        "judge:next", "judge:llava15", "posthoc", "report"])
            self.assertEqual(d.estimate("judge:next"), 0.0)      # no identity record: no judging is planned
            (d.res / "gates" / "next").mkdir(parents=True)
            (d.res / "gates" / "next" / "identity.json").write_text(json.dumps({"judge_needed": True, "status": "DIFFERENT"}))
            self.assertEqual(d.estimate("judge:next"), self.D.JUDGE_SECONDS)
            self.assertIs(d.judge_needed("next"), True)
            self.assertIsNone(d.judge_needed("llava15"))
            (d.res / "gates" / "next" / "admission.json").write_text(json.dumps({"status": "FAIL"}))
            self.assertEqual(d.estimate("main:next"), 0.0)      # a model that was not admitted has no main phase left

    def test_judge_is_the_main_experiment_judge_for_one_method(self):
        text = (ROOT / "scripts/235_convbench_hyperinfer_judge.py").read_text()
        self.assertIn('"219_convbench_main_judge.py"', text)
        self.assertIn("reuse_alignment_cache=False", text)
        J = HC.load_with_common("_convbench_hyperinfer_judge_base", "219_convbench_main_judge.py")
        self.assertIs(J.C, C)
        self.assertEqual(J.C.MAIN_ROLES, [HC.HYPER])
        self.assertEqual(J.C.JUDGE_CAP, 2048)
        self.assertEqual(J.plan_stages(["a", "b", "c"]), [("_first_turn", "judge"), ("_second_turn", "judge"), ("_third_turn", "judge")])

    def test_driver_script_detaches_and_locks(self):
        text = (ROOT / "scripts/run_convbench_hyperinfer.sh").read_text()
        for needle in ("setsid nohup", "flock -n 9", "driver.pid", "< /dev/null", "234_convbench_hyperinfer_driver.py"):
            self.assertIn(needle, text)
        self.assertNotIn("read -", text)      # no point that waits for input


class Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = _load("_test_convbench_hyperinfer_report", "233_report_convbench_hyperinfer.py")

    def test_paired_ttft(self):
        R = self.R
        a = {"c1": (100.0, 120.0), "c2": (110.0, 130.0), "c3": (90.0, 95.0)}
        b = {"c1": (200.0, 240.0), "c2": (220.0, 260.0), "c9": (1.0, 1.0)}
        p = R.paired_ttft(a, b)
        self.assertEqual(p["conversations"], 2)
        self.assertAlmostEqual(p["ratio"]["ratio"], 0.5)
        self.assertAlmostEqual(p["diff"]["mean"], -115.0)
        self.assertAlmostEqual(p["by_turn"]["T2"]["diff_ms"], -105.0)
        self.assertIsNone(R.paired_ttft(a, {}))

    def test_impress_stats(self):
        s = self.R.impress_stats([{"fb": [0, 1], "sim": [0.5, 0.1], "kc": [2, 8], "vc": [2, 8], "nc": 8},
                                  {"fb": [0, 0], "sim": [0.5, 0.5], "kc": [4, 4], "vc": [4, 4], "nc": 8}])
        self.assertAlmostEqual(s["fallback_rate"], 0.25)
        self.assertAlmostEqual(s["mean_jaccard"], 0.4)
        self.assertAlmostEqual(s["kv_chunk_fraction"], (0.25 + 1 + 0.5 + 0.5) / 4)
        self.assertAlmostEqual(s["kv_chunk_fraction_non_fallback_layers"], (0.25 + 0.5 + 0.5) / 3)
        self.assertIsNone(self.R.impress_stats([]))

    def test_length_bins(self):
        R = self.R
        cells, rec = [], {}
        for i in range(40):
            hl = {k: np.array([np.nan, 1.0 + i, 1.0 + i], dtype=np.float32) for k in ("window_ms", "reads_done_in_window", "chunks_done_in_window", "predicted_chunks")}
            cells.append({"cid": f"c{i}", "turn": 2 + i % 2, "suffix": 100 + 10 * i, "ttft": 500.0 + i, "hl": hl,
                          "h": {"hit_tokens": i, "used_tokens_after_layer0": 100}})
            rec[(f"c{i}", 2 + i % 2)] = 1000.0 + 2 * i
        out = R.length_bins(cells, rec)
        self.assertEqual([b["requests"] for b in out["bins"]], [10, 10, 10, 10])
        self.assertEqual((out["bins"][0]["suffix_min"], out["bins"][3]["suffix_max"]), (100, 490))
        self.assertLess(out["bins"][0]["window_ms_per_layer"], out["bins"][3]["window_ms_per_layer"])
        self.assertAlmostEqual(out["bins"][0]["coverage"], sum(range(10)) / 1000)
        self.assertAlmostEqual(out["bins"][0]["ratio"], sum(500.0 + i for i in range(10)) / sum(1000.0 + 2 * i for i in range(10)))
        self.assertIsNone(R.length_bins(cells[:3], rec))

    def test_plain_json(self):
        out = self.R.plain({"a": np.float32(1.5), "b": [np.int64(2), float("nan")], "c": np.array([1.0, np.inf]), "name": "HyperInfer, 읽지 않은 행 NaN"})
        self.assertEqual(out, {"a": 1.5, "b": [2, None], "c": [1.0, None], "name": "HyperInfer, 읽지 않은 행 NaN"})
        json.dumps(out, allow_nan=False)

    def test_quality_uses_recorded_impress_judgments_when_tokens_are_identical(self):
        import tempfile
        R = self.R
        with tempfile.TemporaryDirectory() as tmp:
            q = R.quality(tmp, "llava15", {"judge_needed": False})
            self.assertEqual(q["source"], "recorded_impress")
            self.assertEqual(q["hyperinfer"], q["recorded"]["impress"])
            self.assertEqual([e["pair"] for e in q["pairs"]], ["Ours − HyperInfer", "HyperInfer − FullLoad", "HyperInfer − ReComp"])
            self.assertTrue(all(e["n"] == 577 for e in q["pairs"]))
            self.assertIsNone(R.quality(tmp, "llava15", None)["hyperinfer"])                       # no comparison yet: no score
            self.assertIsNone(R.quality(tmp, "llava15", {"judge_needed": True})["hyperinfer"])     # judging needed and not done: no score

    def test_report_names_only_the_five_methods(self):
        text = (ROOT / "scripts/233_report_convbench_hyperinfer.py").read_text()
        for name in ("hyperinfer_async", "impress_official_async", "OfficialImpressSelector", "HyperInferSelector", "kv_prefetch"):
            self.assertNotIn(name, text)


if __name__ == "__main__":
    unittest.main()

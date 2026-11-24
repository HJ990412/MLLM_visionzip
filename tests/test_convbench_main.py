"""CPU contracts of the ConvBench main experiment harness (docs/convbench_main_contract.md;
scripts 217 / 218 / 219 / 220 / 221, scripts/convbench_main_common.py).

No model is loaded.  What is pinned:

* the arm table and the phase lists (four methods, nine gate arms, rotation balance, 4 / 40 / 577);
* the imported reference harnesses: every role maps to an arm of scripts/209 (LLaVA-NeXT) and
  scripts/206 (LLaVA-1.5) with the expected family, store kind and read mode;
* the frozen ConvBench settings carried over from the alignment: the prompt renderer, the cap, the
  official-token-id check on both tokenizers;
* the resume bookkeeping (interrupted conversation -> discarded journal, removed stores, log line)
  and the guard that deletion never leaves the run's scratch area;
* the gate comparisons, the judge stage plan and the report statistics on synthetic rows.
"""
from __future__ import annotations

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

import convbench_main_common as C  # noqa: E402


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(role, turn, tokens, status="ok", **extra):
    base = {"role": role, "method": C.DISPLAY[role], "turn_id": turn, "status": status,
            "generated_token_ids": list(tokens) if status == "ok" else None, "generated_tokens": len(tokens) if status == "ok" else None,
            "prediction": " ".join(map(str, tokens)) if status == "ok" else None, "first_token_id": tokens[0] if tokens else None,
            "ssd_read_bytes": 0 if turn == 1 or role == C.RECOMP else 1000, "read_calls": 0 if turn == 1 or role == C.RECOMP else 3,
            "ttft_ms": 100.0, "request_e2e_ms": 200.0, "prefill_ms": 90.0, "hook_ms": 30.0, "scatter_ms": None,
            "read_blocked_ms": None if turn == 1 or role == C.RECOMP else 10.0, "input_tokens": 1000 + 100 * turn,
            "input_above_4096": False, "cap_reached": False}
    base.update(extra)
    return base


def _doc(cid, roles, tokens_of, **extra):
    rows = [_row(role, t, tokens_of(role, t)) for t in (1, 2, 3) for role in roles]
    doc = {"conversation_id": cid, "rows": rows, "failures": [], "stores": {}, "roles": roles, "seconds": 10.0,
           "t1_tokens_identical_main_methods": True}
    doc.update(extra)
    return doc


class ArmsAndPhases(unittest.TestCase):
    def test_four_methods_and_nine_gate_arms(self):
        self.assertEqual([C.DISPLAY[r] for r in C.MAIN_ROLES], ["ReComp", "FullLoad", "IMPRESS", "Ours"])
        self.assertEqual(len(C.GATE_ROLES), 9)
        self.assertTrue(set(C.MAIN_ROLES) <= set(C.GATE_ROLES))
        for phase in ("pilot", "overflow", "main"):
            self.assertEqual(C.PHASE_ROLES[phase], C.MAIN_ROLES)

    def test_phase_lists(self):
        index, _ = C.load_index()
        self.assertEqual(len(index), 577)
        self.assertEqual(len(C.GATE_ORDINALS), 4)
        self.assertEqual(len(set(C.PILOT_ORDINALS)), 40)
        self.assertTrue(all(0 <= o < 577 for o in C.GATE_ORDINALS + C.PILOT_ORDINALS))
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp) / "convbench_main_x"
            self.assertEqual(len(C.phase_conversations(res, "next", "main", index)), 577)
            self.assertEqual(len(C.phase_conversations(res, "next", "pilot", index)), 40)
            self.assertEqual(len(C.phase_conversations(res, "next", "gate", index)), 4)
            self.assertIsNone(C.phase_conversations(res, "next", "overflow", index))      # selected only after the pilot

    def test_rotation_is_balanced_over_the_main_phase(self):
        counts = {r: [0, 0, 0, 0] for r in C.MAIN_ROLES}
        for o in range(577):
            order = C.rotate(C.MAIN_ROLES, o)
            self.assertEqual(sorted(order), sorted(C.MAIN_ROLES))
            for pos, r in enumerate(order):
                counts[r][pos] += 1
        for r in C.MAIN_ROLES:
            self.assertLessEqual(max(counts[r]) - min(counts[r]), 1)

    def test_expected_request_count(self):
        self.assertEqual(2 * 577 * 3 * len(C.MAIN_ROLES), 13848)

    def test_safe_name(self):
        self.assertEqual(C.safe_name("convbench:12"), "convbench_12")
        with self.assertRaises(ValueError):
            C.safe_name("..")


class ReferenceHarness(unittest.TestCase):
    """Every role is an arm of the imported reference script, with the family / kind / read mode the contract names."""

    @classmethod
    def setUpClass(cls):
        cls.E = _load("_test_convbench_main_eval", "217_eval_convbench_main.py")

    def _check(self, model, ours_kind_prefix, vector):
        be = self.E.Backend(model)
        B = be.B
        expect = {C.RECOMP: ("recompute", False), C.FULL: ("fullload", True), C.IMP: ("impress_official", True), C.OURS: ("ours", True),
                  C.IMP_SYNC: ("impress_official", False), C.IMP_R100: ("impress_official", False), C.FULL_ON_IMP: ("fullload", False),
                  C.OURS_SYNC: ("ours", False), C.OURS_REPEAT: ("ours", True)}
        for role, (family, is_async) in expect.items():
            arm = B.ARMS[be.arms[role]]
            self.assertEqual(arm["family"], family, role)
            self.assertEqual(bool(arm["async"]), is_async, role)
        self.assertEqual(B.ARMS[be.arms[C.IMP]]["ratio"], 0.25)
        self.assertEqual(B.ARMS[be.arms[C.IMP_R100]]["ratio"], 1.0)
        self.assertEqual(be.kind(C.RECOMP), "none")
        self.assertEqual(be.kind(C.FULL), "raster")
        self.assertEqual(be.kind(C.IMP), be.kind(C.IMP_SYNC))
        self.assertEqual(be.kind(C.IMP), be.kind(C.IMP_R100))
        self.assertEqual(be.kind(C.FULL_ON_IMP), be.kind(C.IMP))      # the FullLoad loader reads the SAME reordered store
        self.assertEqual(be.kind(C.OURS), f"{ours_kind_prefix}{vector}")
        self.assertEqual(be.kind(C.OURS), be.kind(C.OURS_SYNC))
        self.assertEqual(be.kind(C.OURS), be.kind(C.OURS_REPEAT))
        self.assertEqual(B.VECTOR, vector)
        self.assertEqual(B.DATASET, "ConvBench")
        self.assertEqual(len({be.kind(r) for r in C.MAIN_ROLES}), 4)      # no store, raster, IMPRESS, Ours
        return be

    def test_llava_next_roles(self):
        be = self._check("next", "packedorig_", "PYR-beta-raw")
        self.assertEqual(be.B.READAHEAD[be.arms[C.FULL]], "layer_stream")
        self.assertEqual(be.B.READAHEAD[be.arms[C.IMP]], "next_layer")
        self.assertEqual(be.B.READAHEAD[be.arms[C.OURS]], "layer_stream")
        self.assertEqual(be.reference["fullload_impress"], "results/mt_gqa_impress_code_20261117T185832Z")
        self.assertEqual(be.reference["ours"], "results/mt_gqa_ours_order_20261120T052725Z")

    def test_llava15_roles(self):
        be = self._check("llava15", "packedorig_", "PYR-b20")
        self.assertEqual(be.B.READAHEAD[be.arms[C.FULL]], "layer_stream")
        self.assertEqual(be.B.READAHEAD[be.arms[C.IMP]], "next_layer")
        self.assertEqual(be.B.READAHEAD[be.arms[C.OURS]], "layer_stream")
        self.assertEqual(be.reference["ours"], "results/llava15_mt_gqa_main_20261120T165001Z")

    def test_reference_checks_pass_and_source_is_unchanged(self):
        for model in C.MODELS:
            out = self.E.reference_checks(self.E.Backend(model))
            latest = out["llava_next_mtvqa_main_20261121" if model == "next" else "llava15_mt_gqa_main_20261120"]
            self.assertEqual(latest["changed"], [])

    def test_ours_budget_documents(self):
        for model in C.MODELS:
            be = self.E.Backend(model)
            doc = be.docs()[be.B.VECTOR]
            self.assertEqual(doc["layer_budget_sha256"], be.B.VECTOR_RATIO_SHA)
            self.assertEqual(float(doc["total_ratio"]), 0.25)
            self.assertEqual(len(doc["layer_keep_ratios"]), 32)


class FrozenSettings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = _load("_test_convbench_main_eval", "217_eval_convbench_main.py")
        cls.G = cls.E.gen_module()

    def test_cap_is_always_1024(self):
        self.assertEqual(C.MAX_NEW, 1024)
        self.assertEqual(self.G.NOMINAL_CAP, 1024)
        self.assertEqual(self.G.effective_cap("official_1024", 4000, 4096), (1024, False))

    def test_prompt_rendering(self):
        q = ["What is this?", "Why?", "Write a slogan."]
        p1 = self.E.render(self.G, "llava15", q[:1], [])
        self.assertEqual(p1, f"{self.G.VICUNA_SYSTEM} USER: <image>\nWhat is this? ASSISTANT:")
        p3 = self.E.render(self.G, "llava15", q, ["A car.", "Because."])
        self.assertEqual(p3, f"{self.G.VICUNA_SYSTEM} USER: <image>\nWhat is this? ASSISTANT: A car.</s>USER: Why? ASSISTANT: Because.</s>USER: Write a slogan. ASSISTANT:")
        self.assertEqual(p3, self.G.canonical_prompt(q, ["A car.", "Because."]))
        n3 = self.E.render(self.G, "next", q, ["A car.", "Because."])
        self.assertIn("<image> \n", n3)
        self.assertIn("</s> USER: Why?", n3)
        self.assertEqual(n3.replace("<image> \n", "<image>\n").replace("</s> USER", "</s>USER"), p3)

    def test_official_token_check_on_both_tokenizers(self):
        from transformers import AutoTokenizer
        hub = Path.home() / ".cache/huggingface/hub"
        sources = {"next": "llava-hf/llava-v1.6-vicuna-7b-hf", "llava15": str(sorted(hub.glob("models--llava-hf--llava-1.5-7b-hf/snapshots/*"))[0])}
        q = ["What texts are written on the car?", "Answer why it looks like that.", "Create a slogan for the car."]
        a = ["The car says HUMMER.", "It is a large SUV.\n\nIt has a custom plate."]
        for model, source in sources.items():
            fast = AutoTokenizer.from_pretrained(source)
            slow = AutoTokenizer.from_pretrained(source, use_fast=False, legacy=True)
            image_id = fast.convert_tokens_to_ids("<image>")
            for turn in (1, 2, 3):
                prompt = self.E.render(self.G, model, q[:turn], a[:turn - 1])
                canonical = self.G.canonical_prompt(q[:turn], a[:turn - 1])
                ok, n = self.E.token_check(self.G, fast, slow, image_id, prompt, canonical)
                self.assertTrue(ok, (model, turn))
                self.assertGreater(n, 30)
            # the September rendering of LLaVA-NeXT does NOT give the official ids (the defect the alignment fixed)
            if model == "next":
                sept = self.G.render_prompt(q, a, *self.G.SEPT_RENDER)
                self.assertFalse(self.E.token_check(self.G, fast, slow, image_id, sept, self.G.canonical_prompt(q, a))[0])
            # an answer that contains an image marker cannot be used as history
            bad = self.E.render(self.G, model, q[:2], ["look at <image> here"])
            self.assertFalse(self.E.token_check(self.G, fast, slow, image_id, bad, self.G.canonical_prompt(q[:2], ["look at <image> here"]))[0])


class Lifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = _load("_test_convbench_main_eval", "217_eval_convbench_main.py")

    def test_remove_tree_only_inside_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp) / "convbench_main_x"
            store = res / "scratch" / "next" / "main" / "convbench_1" / "raster"
            store.mkdir(parents=True)
            (store / "k.bin").write_bytes(b"x" * 100)
            other = res / "next" / "main"
            other.mkdir(parents=True)
            (other / "keep.json").write_text("{}")
            with self.assertRaises(ValueError):
                self.E.remove_tree(res, other)
            with self.assertRaises(ValueError):
                self.E.remove_tree(res, res / "scratch")
            self.assertEqual(self.E.remove_tree(res, store.parent), 100)
            self.assertFalse(store.parent.exists())
            self.assertTrue((other / "keep.json").exists())
            self.assertEqual(self.E.remove_tree(res, store.parent), 0)

    def test_interrupted_conversation_is_discarded_and_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp) / "convbench_main_x"
            pdir = C.phase_dir(res, "next", "main")
            (pdir / "partial").mkdir(parents=True)
            journal = pdir / "partial" / "convbench_7.attempt1.jsonl"
            for t in (1, 1, 1, 1, 2):
                C.append_jsonl(journal, {"event": "row", "turn_id": t})
            store = res / "scratch" / "next" / "main" / "convbench_7" / "raster"
            store.mkdir(parents=True)
            (store / "k.bin").write_bytes(b"x" * 64)
            event = self.E.discard_partial(res, "next", "main", "convbench:7", 1, "interrupted_conversation_restarts_at_turn1")
            self.assertEqual(event["rows_discarded"], 5)
            self.assertEqual(event["store_bytes_removed"], 64)
            self.assertFalse(journal.exists())
            self.assertFalse(store.exists())
            self.assertEqual(len(list((pdir / "discarded").iterdir())), 1)
            log = C.read_jsonl(res / "resume_log.jsonl")
            self.assertEqual(len(log), 1)
            self.assertEqual(log[0]["conversation_id"], "convbench:7")
            self.assertTrue(log[0]["mode"].startswith("interrupted"))

    def test_failed_row_shape(self):
        be = mock.Mock()
        be.arms = {C.IMP: "impress_official_async"}
        be.family.return_value = "impress_official"
        row = self.E.failed_row(be, C.IMP, 3, [C.RECOMP, C.FULL, C.IMP, C.OURS], "OutOfMemoryError", executed=True)
        self.assertEqual((row["status"], row["method"], row["turn_id"], row["prediction"]), ("failed", "IMPRESS", 3, None))

    def test_freeze_violations_ignore_new_files_and_report_scripts(self):
        now = C.source_hashes()
        self.assertFalse(any(C.REPORT_SCRIPT.match(k) for k in now))
        self.assertIn("scripts/217_eval_convbench_main.py", now)
        self.assertNotIn("scripts/220_report_convbench_main.py", now)
        frozen = dict(list(now.items())[:5])
        self.assertEqual(C.freeze_violations(frozen), [])
        frozen["mmimpress/does_not_exist.py"] = "0" * 64
        self.assertEqual(C.freeze_violations(frozen), ["mmimpress/does_not_exist.py"])

    def test_atomic_gz_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "x.json.gz"
            C.atomic_gz_json(path, {"한글": [1, 2, 3], "x": None})
            self.assertEqual(C.read_gz_json(path), {"한글": [1, 2, 3], "x": None})


class Gates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V = _load("_test_convbench_main_validate", "218_validate_convbench_main.py")

    def test_pair_comparison(self):
        same = {"c1": _doc("c1", C.GATE_ROLES, lambda role, t: [t, 5, 2])}
        r = self.V.compare_pair(same, C.OURS_SYNC, C.OURS, (1, 2, 3), True)
        self.assertTrue(r["ok"])
        self.assertEqual(r["requests"], 3)
        diff = {"c1": _doc("c1", C.GATE_ROLES, lambda role, t: [t, 5, 2] if not (role == C.OURS and t == 3) else [t, 6, 2])}
        r = self.V.compare_pair(diff, C.OURS_SYNC, C.OURS, (1, 2, 3), True)
        self.assertFalse(r["ok"])
        self.assertEqual((r["token_mismatches"], r["examples"][0]["first_differing_position"]), (1, 1))
        # read bytes differ on a follow-up
        doc = _doc("c1", C.GATE_ROLES, lambda role, t: [t, 5, 2])
        for row in doc["rows"]:
            if row["role"] == C.IMP and row["turn_id"] == 2:
                row["ssd_read_bytes"] = 999
        r = self.V.compare_pair({"c1": doc}, C.IMP_SYNC, C.IMP, (1, 2, 3), True)
        self.assertEqual((r["ok"], r["byte_mismatches"], r["token_mismatches"]), (False, 1, 0))

    def test_failed_request_is_never_a_match(self):
        doc = _doc("c1", C.GATE_ROLES, lambda role, t: [1, 2])
        for i, row in enumerate(doc["rows"]):
            if row["role"] == C.OURS_REPEAT and row["turn_id"] == 3:
                doc["rows"][i] = _row(C.OURS_REPEAT, 3, [], status="failed")
        r = self.V.compare_pair({"c1": doc}, C.OURS, C.OURS_REPEAT, (1, 2, 3), True)
        self.assertEqual((r["ok"], r["missing"]), (False, 1))

    def test_gate5_turn1(self):
        ok = {"c1": _doc("c1", C.MAIN_ROLES, lambda role, t: [9, 9] if t == 1 else [hash(role) % 7])}
        self.assertTrue(self.V.gate5_t1(ok)["ok"])
        bad = {"c1": _doc("c1", C.MAIN_ROLES, lambda role, t: [9, 9] if role != C.IMP else [9, 8])}
        self.assertEqual(self.V.gate5_t1(bad)["differing"], 1)

    def test_gate2_against_alignment_record(self):
        index, _ = C.load_index()
        conv = index[C.GATE_ORDINALS[0]]
        ref = C.alignment_rows("next", conv["conversation_id"])
        doc = {"conversation_id": conv["conversation_id"], "failures": [], "rows": [
            _row(C.RECOMP, t, [0], prediction=ref[t - 1]["prediction"], generated_tokens=ref[t - 1]["generated_tokens"],
                 first_token_id=ref[t - 1]["first_token_id"], input_tokens=ref[t - 1]["input_tokens"]) for t in (1, 2, 3)]}
        self.assertTrue(self.V.gate2_recomp("next", {conv["conversation_id"]: doc})["ok"])
        doc["rows"][2]["prediction"] += " extra"
        out = self.V.gate2_recomp("next", {conv["conversation_id"]: doc})
        self.assertEqual((out["ok"], out["mismatches"]), (False, 1))

    def test_gate_result_is_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gates" / "x.json"
            self.V.write_once(path, {"status": "FAIL"}, {"_file_sha256": "a"})
            with self.assertRaises(FileExistsError):
                self.V.write_once(path, {"status": "PASS"}, {"_file_sha256": "a"})
            self.assertEqual(json.loads(path.read_text())["status"], "FAIL")

    def test_baseline_failures_are_the_known_ones(self):
        self.assertEqual(len(self.V.BASELINE_FAILURES["4"]), 3)
        self.assertTrue(all("glm46v" in f for f in self.V.BASELINE_FAILURES["4"]))


class JudgeAndReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.J = _load("_test_convbench_main_judge", "219_convbench_main_judge.py")
        cls.R = _load("_test_convbench_main_report", "220_report_convbench_main.py")

    def test_stage_plan(self):
        P = self.J.plan_stages
        self.assertEqual([a for _, a in P(["a", "b", "c"])], ["judge", "judge", "judge"])
        self.assertEqual([a for _, a in P(["a", "b", None])], ["judge_with_missing_later_answer", "judge_with_missing_later_answer", "generation_failed"])
        self.assertEqual([a for _, a in P(["a", None, None])], ["judge_with_missing_later_answer", "generation_failed", "generation_failed"])
        self.assertEqual([a for _, a in P([None, None, None])], ["generation_failed"] * 3)

    def test_judge_uses_the_frozen_settings(self):
        J214 = self.J.load_214()
        self.assertEqual(C.JUDGE_CAP, 2048)
        self.assertEqual((J214.OFFICIAL_CAP, J214.EXTRACTION_CAP), (512, 512))
        self.assertTrue(str(J214.CHECKPOINT).endswith("0e9e39f249a16976918f6564b8830bc894c89659"))
        self.assertEqual(J214.parse_view("Overall, Response A is better.", None), {"winner": "A", "parse_stage": "first_pass"})
        self.assertEqual(J214.parse_view("no verdict", {"text": "Final Answer: Response B"}), {"winner": "B", "parse_stage": "second_pass"})
        self.assertEqual(J214.parse_view("no verdict", {"text": "unclear"})["parse_stage"], "unresolved")

    def test_bootstrap_is_seeded_and_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        v = rng.normal(1.0, 2.0, size=577)
        a = self.R.bootstrap_mean(v); b = self.R.bootstrap_mean(v)
        self.assertEqual(a, b)
        self.assertLess(a["ci_low"], a["mean"]); self.assertLess(a["mean"], a["ci_high"])
        self.assertAlmostEqual(a["mean"], float(v.mean()))
        r = self.R.bootstrap_ratio(np.full(50, 2.0), np.full(50, 4.0))
        self.assertEqual((r["ratio"], r["ci_low"], r["ci_high"]), (0.5, 0.5, 0.5))

    def test_system_tables_on_synthetic_rows(self):
        docs = {}
        for i in range(6):
            doc = _doc(f"c{i}", C.MAIN_ROLES, lambda role, t: [1, 2, 3])
            for row in doc["rows"]:
                if row["turn_id"] > 1:
                    row["ttft_ms"] = {C.RECOMP: 500.0, C.FULL: 400.0, C.IMP: 800.0, C.OURS: 200.0}[row["role"]] + i
                    if row["role"] == C.IMP:
                        row["impress"] = {"fb": [0, 1], "sim": [0.5, 0.1], "kc": [2, 4], "vc": [2, 4], "nc": 4, "raster": [4, 4], "only": None,
                                          "share": [0.9, 0.8]}
            docs[f"c{i}"] = doc
        t = self.R.system_tables(docs)
        self.assertAlmostEqual(t["roles"][C.OURS]["ttft"]["T2-T3"]["mean"], 202.5)
        ratio = {e["ratio_name"]: e for e in t["ttft_ratios"]}
        self.assertAlmostEqual(ratio["Ours / FullLoad"]["ratio"], 202.5 / 402.5)
        pair = {e["pair"]: e for e in t["ttft_pairs"]}
        self.assertAlmostEqual(pair["Ours − ReComp"]["mean"], -300.0)
        self.assertEqual(t["impress"]["fallback_rate"], 0.5)
        self.assertAlmostEqual(t["impress"]["kv_chunk_fraction"], 0.75)
        self.assertAlmostEqual(t["impress"]["kv_chunk_fraction_non_fallback_layers"], 0.5)
        self.assertAlmostEqual(t["impress"]["raster_chunk_fraction_computed"], 1.0)
        self.assertEqual(t["roles"][C.RECOMP]["ssd_mb_per_followup"], 0.0)
        self.assertEqual(t["failed_rows"], [])
        # a failed follow-up leaves the conversation out of the pairs that involve the method, and is listed
        docs["c0"]["rows"] = [r if not (r["role"] == C.IMP and r["turn_id"] == 3) else _row(C.IMP, 3, [], status="failed", failure="OOM", executed=True)
                              for r in docs["c0"]["rows"]]
        t2 = self.R.system_tables(docs)
        pair2 = {e["pair"]: e for e in t2["ttft_pairs"]}
        self.assertEqual((pair2["Ours − IMPRESS"]["n"], pair2["Ours − FullLoad"]["n"]), (5, 6))
        self.assertEqual(len(t2["failed_rows"]), 1)
        md = "\n".join(self.R.system_md(t2))
        for name in ("ReComp", "FullLoad", "IMPRESS", "Ours"):
            self.assertIn(name, md)
        for forbidden in ("impress_official", "packedorig", "fullload_async", "PackedImageContext"):
            self.assertNotIn(forbidden, md)

    def test_report_names_only_the_four_methods(self):
        text = "\n".join(self.R.DEVIATIONS + self.R.LIMITS)
        for forbidden in ("impress_official", "packedorig", "fullload_async", "request_packed"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()

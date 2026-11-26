"""CPU contracts of the LLaVA-NeXT x MT-GQA main experiment v2
(docs/mt_gqa_main_v2_contract.md; mmimpress/mt_gqa_tokens.py, scripts 222 / 223 / 224, scripts/run_mt_gqa_main_v2.sh).

No model is loaded.  What is pinned:

* the token-sequence option: "legacy" hands the tokenizer the prompt itself, "official" the prompt with one
  space after ``<image>``; with the real tokenizers of the pinned checkpoints, the official path emits the
  original-LLaVA ids, the legacy path lacks exactly one token right after the image, and the LLaVA-1.5
  tokenizer needs no rendering;
* the harness: phases, arms and request counts; the method calls are the calls of the reference harness;
  the prompt string is unchanged; the dialogue-level resume bookkeeping;
* the gate comparisons and the report's tables on synthetic rows;
* the driver: detached, locked, no prompt for input.
"""
from __future__ import annotations

import ast
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmimpress import mt_gqa_tokens as TOK  # noqa: E402

NEXT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
L15_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
SHORT = "Answer the current question with a single word or short phrase."


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


def _transformers_major():
    import transformers
    return int(transformers.__version__.split(".")[0])


def _prompts():
    """MT-GQA prompts of turns 1-3 as scripts/156 ``prompt_factory`` builds them."""
    q = ["Is the sky blue?", "What color is the car on the left?", "Are there any people?"]
    answers = ["Yes", "it's red", "No."]
    out = []
    for turn in (1, 2, 3):
        lines = []
        for t in range(1, turn):
            lines += [f"Q{t}: {q[t - 1]}", f"A{t}: {answers[t - 1]}"]
        hist = "\n".join(lines)
        body = ([hist, ""] if hist else []) + [f"Current question Q{turn}: {q[turn - 1]}", f"{SHORT} ASSISTANT:"]
        out.append("USER: <image>\n" + "\n".join(body))
    return out


class TokenModuleTests(unittest.TestCase):
    def test_legacy_is_the_prompt_and_official_adds_one_space_after_the_image(self):
        for prompt in _prompts():
            self.assertEqual(TOK.tokenizer_input(prompt, "legacy"), prompt)
            text = TOK.tokenizer_input(prompt, "official")
            self.assertEqual(len(text), len(prompt) + 1)
            self.assertEqual(text, prompt.replace("<image>\n", "<image> \n"))
            self.assertEqual(TOK.prompt_of(text, "official"), prompt)
            self.assertEqual(TOK.prompt_of(prompt, "legacy"), prompt)

    def test_official_refuses_prompts_the_rule_was_not_checked_for(self):
        for bad in ("USER: no image\nQ ASSISTANT:", "USER: <image>\nA <image>\nB ASSISTANT:", "USER: <image> Q ASSISTANT:",
                    "USER: <image>\nQ ASSISTANT: yes</s>USER: Q2 ASSISTANT:"):
            with self.assertRaises(ValueError):
                TOK.tokenizer_input(bad, "official")
        with self.assertRaises(ValueError):
            TOK.tokenizer_input(_prompts()[0], "something")
        with self.assertRaises(ValueError):
            TOK.prompt_of(_prompts()[0], "official")

    def test_collapse_image(self):
        self.assertEqual(TOK.collapse_image([1, 7, 9, 9, 9, 5, 9, 6], 9), [1, 7, -200, 5, -200, 6])
        self.assertEqual(TOK.collapse_image([1, 7], 9), [1, 7])

    def test_after_image_text_is_the_convbench_rendering(self):
        src = (ROOT / "scripts/213_convbench_recomp_generate.py").read_text()
        table = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "OFFICIAL_RENDER")
        self.assertEqual(table["next"][0], TOK.OFFICIAL_AFTER_IMAGE + "\n")
        self.assertEqual(table["llava15"][0], "\n")           # LLaVA-1.5 needs no rendering


@unittest.skipUnless(_transformers_major() == 4 and NEXT_SNAPSHOT.is_dir(), "needs the slow legacy tokenizer (transformers 4.x) and the pinned LLaVA-NeXT snapshot")
class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fast, cls.slow, cls.image = TOK.load_tokenizers(str(NEXT_SNAPSHOT))

    def test_official_path_emits_the_original_llava_ids(self):
        for prompt in _prompts():
            self.assertTrue(TOK.compare(self.fast, self.slow, self.image, prompt, "official")["match"])

    def test_legacy_path_lacks_exactly_one_token_right_after_the_image(self):
        space = self.fast.convert_tokens_to_ids("▁")
        for prompt in _prompts():
            got = TOK.compare(self.fast, self.slow, self.image, prompt, "legacy")
            self.assertFalse(got["match"])
            legacy, official = got["path_ids"], got["official_ids"]
            at = legacy.index(TOK.OFFICIAL_IMAGE_INDEX)
            self.assertEqual(official, legacy[:at + 1] + [space] + legacy[at + 1:])
            # the ids up to and including the image are the same: the stored visual prefix covers the same positions
            self.assertEqual(official[:at + 1], legacy[:at + 1])

    def test_original_procedure_is_the_one_of_the_convbench_alignment(self):
        G = _load("_test_mtgqa_v2_213", "213_convbench_recomp_generate.py")
        for prompt in _prompts():
            self.assertEqual(TOK.official_token_ids(self.slow, prompt), G.official_token_ids(self.slow, prompt))
            ids = self.fast(prompt).input_ids
            self.assertEqual(TOK.collapse_image(ids, self.image), G.collapse_image(ids, self.image))

    def test_odd_generated_answers_in_the_history(self):
        base = _prompts()[1]
        for answer in ("", " yes", "yes ", "a\nb", "  ", "No.", "it's 3", "left side", "two (2)"):
            prompt = base.replace("A1: Yes", "A1: " + answer)
            self.assertTrue(TOK.compare(self.fast, self.slow, self.image, prompt, "official")["match"], repr(answer))

    @unittest.skipUnless(L15_SNAPSHOT.is_dir(), "needs the pinned LLaVA-1.5 snapshot")
    def test_llava15_tokenizer_emits_the_original_ids_without_rendering(self):
        fast, slow, image = TOK.load_tokenizers(str(L15_SNAPSHOT))
        for prompt in _prompts():
            self.assertTrue(TOK.compare(fast, slow, image, prompt, "legacy")["match"])


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H = _load("_test_mtgqa_main_v2_eval", "222_eval_mt_gqa_main_v2.py")

    def test_phases_arms_and_request_counts(self):
        H = self.H
        self.assertEqual(H.MAIN_METHODS, ["recompute", "fullload_async", "impress_official_async", "ours_origorder_kv25_PYR-beta-raw_packed_async"])
        self.assertEqual(H.PHASE_METHODS["main"], H.MAIN_METHODS)
        self.assertEqual(H.PHASE_METHODS["dev"], H.MAIN_METHODS)
        self.assertEqual(H.PHASE_METHODS["legacy"], H.MAIN_METHODS)
        self.assertEqual(len(H.PHASE_METHODS["gate"]), 9)
        self.assertEqual(set(H.PHASE_METHODS["gate"]) - set(H.MAIN_METHODS), set(H.GATE_ONLY))
        self.assertEqual(set(H.PHASE_METHODS["devgate"]), set(H.GATE_ONLY) | {H.NEW_ASYNC, H.ORIG_ASYNC})
        self.assertEqual(H.EXPECTED_REQUESTS, {"legacy": 588, "gate": 1323, "devgate": 7938, "dev": 4536, "main": 48732})
        self.assertEqual(H.EXPECTED_REQUESTS["main"], 4 * 12183)
        for m in H.MAIN_METHODS[1:]:
            self.assertTrue(H.ARMS[m]["async"], m)           # the three stored-KV methods read asynchronously

    def test_only_the_legacy_phase_uses_the_legacy_tokenization(self):
        H = self.H
        self.assertEqual(H.PHASE_TOKENIZATION, {"legacy": "legacy", "gate": "official", "devgate": "official", "dev": "official", "main": "official"})

    def test_backend_is_the_reference_harness_and_writes_this_runs_identity(self):
        H = self.H
        self.assertTrue(H.N.__file__.endswith("209_eval_llava_next_mtvqa_main.py"))
        self.assertEqual((H.N.DATASET, H.N.SCHEMA), ("MT-GQA-reconstructed", H.SCHEMA))
        self.assertEqual(H.IMPL_REF_RUN.name, "mt_gqa_impress_code_20261117T185832Z")
        self.assertEqual(H.OURS_REF_RUN.name, "mt_gqa_ours_order_20261120T052725Z")
        self.assertEqual(H.VECTOR, "PYR-beta-raw")

    def test_request_writer_and_activation_calls_are_the_backends(self):
        pixel = _function_source("222_eval_mt_gqa_main_v2.py", "pixel_request")
        self.assertIn("N.turn1_request_with_stats(", pixel); self.assertIn("N.normal_request(", pixel)
        image = _function_source("222_eval_mt_gqa_main_v2.py", "run_image")
        for call in ("N.stored_request(", "N.persist(", "N.activate("):
            self.assertIn(call, image)
        # no request function of mmimpress is called directly by the new harness
        src = (ROOT / "scripts/222_eval_mt_gqa_main_v2.py").read_text()
        for name in ("request_packed(", "request_official(", ".recompute(", "persist_captured_"):
            self.assertNotIn(name, src)

    def test_arm_settings_equal_the_reference_runs(self):
        H = self.H
        impl, ours = H.IMPL_REF_RUN / "config.json", H.OURS_REF_RUN / "config.json"
        if not (impl.is_file() and ours.is_file()):
            self.skipTest("reference runs not on this machine")
        impl, ours = json.loads(impl.read_text()), json.loads(ours.read_text())
        strip = lambda a: {k: v for k, v in a.items() if k != "display"}      # noqa: E731
        for m in (H.RECOMP, H.FULL_ASYNC, H.NEW, H.NEW_ASYNC, H.NEW_R100, H.FULL_REORDERED):
            self.assertEqual(strip(impl["arms"][m]), strip(H.ARMS[m]), m)
        for m in (H.ORIG, H.ORIG_ASYNC, H.ORIG_REPEAT):
            self.assertEqual(strip(ours["arms"][m]), strip(H.ARMS[m]), m)
        self.assertEqual(impl["server_args"], H.SERVER_ARGS); self.assertEqual(impl["impress_official"], H.OFFICIAL)

    def test_prompt_factory_keeps_the_prompt_and_renders_the_tokenizer_input(self):
        H = self.H
        dialog = {"dialog_id": "d1", "image_id": "i1", "turns": [{"question_id": 1, "question": "Is it day?"}, {"question_id": 2, "question": "What color?"},
                                                                  {"question_id": 3, "question": "How many?"}]}
        methods = H.MAIN_METHODS
        row = {"method_id": H.RECOMP, "dialog_id": "d1", "prediction": "Yes", "logical_request_id": "L", "physical_execution_id": "P"}
        for mode, after in (("legacy", "<image>\n"), ("official", "<image> \n")):
            f = H.PromptFactory(methods, dialog, 2, H.RECOMP, {1: row}, mode)
            text, history, entries = f()
            self.assertEqual(f.prompt, H.H1.prompt_factory(methods, dialog, 2, H.RECOMP, {1: row})[0])
            self.assertTrue(f.prompt.startswith("USER: <image>\nQ1: Is it day?\nA1: Yes\n\nCurrent question Q2: What color?"))
            self.assertIn(after, text); self.assertEqual(text, TOK.tokenizer_input(f.prompt, mode))
            self.assertEqual(history, "Q1: Is it day?\nA1: Yes")

            class Runner:
                image_token_id = 9
            diag = H.close_diag(Runner(), {"prompt": text, "history": history, "history_entries": entries, "hidden": object(), "vision": object()}, f)
            self.assertEqual((diag["prompt"], diag["tokenizer_input"], diag["tokenization"]), (f.prompt, text, mode))
            self.assertNotIn("hidden", diag); self.assertNotIn("request_ids", diag)

    def test_audit_rebuilds_the_unchanged_prompt_string(self):
        audit = _function_source("222_eval_mt_gqa_main_v2.py", "audit_image_rows")
        self.assertIn('prompt = "USER: <image>\\n" + "\\n".join(body)', audit)
        self.assertIn("request_ids_equal_original_llava_ids", audit)
        self.assertIn("tokenizer_input_is_the_phase_rendering_of_the_prompt", audit)

    def test_complete_prefix_and_torn_journal(self):
        H = self.H
        methods = ["a", "b"]
        dialogs = [{"dialog_id": f"d{i}"} for i in range(3)]
        rows = [{"dialog_id": "d0", "method_id": m, "turn_id": t} for m in methods for t in (1, 2, 3)]
        rows += [{"dialog_id": "d1", "method_id": m, "turn_id": t} for m in methods for t in (1, 2, 3)]
        rows += [{"dialog_id": "d2", "method_id": "a", "turn_id": 1}]
        n, kept, dropped = H.complete_prefix(rows, dialogs, methods)
        self.assertEqual((n, len(kept), len(dropped)), (2, 12, 1))
        n, kept, dropped = H.complete_prefix(rows[1:], dialogs, methods)       # the first dialogue lacks a row: nothing is complete
        self.assertEqual((n, len(kept)), (0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            p.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[1]) + "\n" + '{"dialog_id": "d0", "meth')
            got, torn = H.tolerant_rows(p)
            self.assertEqual((len(got), torn), (2, 1))
            self.assertEqual(H.tolerant_rows(Path(tmp) / "missing.jsonl"), ([], 0))

    def test_plan_image_fresh_continue_and_restart(self):
        H = self.H
        methods = [H.RECOMP, H.ORIG_ASYNC]
        kind = H.ORIG_KIND
        dialogs = [{"dialog_id": "d0"}, {"dialog_id": "d1"}]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertEqual(H.plan_image(run, "main", "img", dialogs, methods)["mode"], "fresh")
            attempt = run / "main" / "images" / "img" / "attempt_0001"; attempt.mkdir(parents=True)
            store = run / "scratch" / "main" / "img" / "attempt_0001" / kind; store.mkdir(parents=True)
            (store / "kv_packed.bin").write_bytes(b"payload"); (store / "integrity.json").write_text("{}")
            hashes = H.file_hashes(store)
            (attempt / f"{kind}_persistence.json").write_text(json.dumps({"files": {"integrity.json": {"sha256": hashes["integrity.json"]}}}))
            (attempt / f"{kind}_hashes.json").write_text(json.dumps({"kv_packed.bin": hashes["kv_packed.bin"]}))
            (attempt / f"{kind}_meta.json").write_text("{}"); (attempt / f"{kind}_activation.json").write_text("{}")
            rows = [{"dialog_id": "d0", "method_id": m, "turn_id": t} for m in methods for t in (1, 2, 3)] + [{"dialog_id": "d1", "method_id": H.RECOMP, "turn_id": 1}]
            (attempt / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            plan = H.plan_image(run, "main", "img", dialogs, methods)
            self.assertEqual((plan["mode"], plan["attempt"], plan["n_complete"], plan["interrupted_dialogue"], plan["restart_dialogues"]),
                             ("continue", "attempt_0001", 1, "d1", ["d1"]))
            self.assertEqual((len(plan["kept_rows"]), len(plan["dropped_rows"])), (6, 1))
            self.assertEqual(plan["rerun_dialogues"], ["d1"])
            (store / "kv_packed.bin").write_bytes(b"changed")                    # the store no longer matches its recorded hash
            plan = H.plan_image(run, "main", "img", dialogs, methods)
            self.assertEqual((plan["mode"], plan["attempt"], plan["restart_dialogues"]), ("fresh", "attempt_0002", ["d0", "d1"]))
            self.assertEqual(plan["rerun_dialogues"], ["d0", "d1"])                  # both had started before the interruption
            (attempt / "raw.jsonl").write_text(json.dumps(rows[0]) + "\n")           # the first dialogue itself was interrupted
            self.assertEqual(H.plan_image(run, "main", "img", dialogs, methods)["mode"], "fresh")
            with self.assertRaises(ValueError):
                H.remove_scratch(run, run / "main")

    def test_no_file_frozen_by_the_earlier_runs_changed_beyond_the_known_edits(self):
        H = self.H
        ref = H.IMPL_REF_RUN / "config.json"
        if not ref.is_file():
            self.skipTest("reference run not on this machine")
        frozen = {k: v for k, v in json.loads(ref.read_text())["source_sha256"].items() if not H.REPORT_SCRIPT.match(k)}
        self.assertLessEqual(set(H.freeze_violations(frozen)), set(H.CHANGED_SINCE_IMPL_REF))
        l15 = ROOT / "runs/llava15_mt_gqa_main_20261120T165001Z/config.json"
        if l15.is_file():      # the LLaVA-1.5 path: nothing its main run froze has changed
            frozen = json.loads(l15.read_text())["source_sha256"]
            self.assertEqual([k for k, v in frozen.items() if not (ROOT / k).is_file() or H.sha(ROOT / k) != v], [])


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = _load("_test_mtgqa_main_v2_gates", "223_validate_mt_gqa_main_v2.py")

    @staticmethod
    def _row(tokens, prompt="p", **extra):
        row = {"result": {"generated_token_ids": tokens}, "prompt": prompt, "image_id": "i", "prediction": "x", "ssd_read_bytes": 10,
               "probe_read_bytes": 1, "k_read_bytes": 2, "v_read_bytes": 3, "packed_read_bytes": 4, "ttft_ms": 1.0,
               "official_token_ids_match": True, "tokenization": "official"}
        row.update(extra)
        return row

    def test_compare_requires_every_request_equal_tokens_and_bytes(self):
        G = self.G
        keys = [("d", 1), ("d", 2)]
        rows = {("a", "d", 1): self._row([1]), ("b", "d", 1): self._row([1]), ("a", "d", 2): self._row([2]), ("b", "d", 2): self._row([2])}
        self.assertEqual(G.compare(rows, "a", "b", keys, ("ssd_read_bytes",))["status"], "PASS")
        rows[("b", "d", 2)] = self._row([3])
        self.assertEqual(G.compare(rows, "a", "b", keys)["tokens_differ"], 1)
        rows[("b", "d", 2)] = self._row([2], ssd_read_bytes=11)
        self.assertEqual(G.compare(rows, "a", "b", keys, ("ssd_read_bytes",))["status"], "FAIL")
        del rows[("b", "d", 2)]
        self.assertEqual(G.compare(rows, "a", "b", keys)["status"], "FAIL")

    def test_turn1_identity_is_judged_on_the_four_methods(self):
        G = self.G
        methods = G.H.MAIN_METHODS
        keys = [("d", 1), ("d", 2), ("d", 3)]
        rows = {(m, "d", 1): self._row([5, 6]) for m in methods}
        self.assertEqual(G.turn1_identity(rows, keys, methods, judged=True)["status"], "PASS")
        rows[(methods[2], "d", 1)] = self._row([5, 7])
        out = G.turn1_identity(rows, keys, methods, judged=True)
        self.assertEqual((out["status"], out["mismatch_count"]), ("FAIL", 1))
        self.assertEqual(G.turn1_identity(rows, keys, methods, judged=False)["status"], "REPORTED")

    def test_gates_4_and_5_pair_the_right_arms(self):
        G = self.G; H = G.H
        keys = [("d", 2)]
        rows = {(m, "d", 2): self._row([1]) for m in H.PHASE_METHODS["devgate"]}
        g4, g5 = G.gates_4_5(rows, keys)
        self.assertEqual((g4["status"], g5["status"]), ("PASS", "PASS"))
        self.assertEqual(g4["sync_vs_async"]["arms"], [H.ORIG, H.ORIG_ASYNC]); self.assertEqual(g4["executed_twice"]["arms"], [H.ORIG_ASYNC, H.ORIG_REPEAT])
        self.assertEqual(g5["retention_100_vs_fullload_loader_on_reordered_store"]["arms"], [H.NEW_R100, H.FULL_REORDERED])
        self.assertEqual(g5["sync_vs_async"]["arms"], [H.NEW, H.NEW_ASYNC])
        rows[(H.ORIG_REPEAT, "d", 2)] = self._row([2])
        self.assertEqual(G.gates_4_5(rows, keys)[0]["status"], "FAIL")
        rows[(H.ORIG_REPEAT, "d", 2)] = self._row([1]); rows[(H.NEW, "d", 2)] = self._row([1], k_read_bytes=99)
        self.assertEqual(G.gates_4_5(rows, keys)[1]["status"], "FAIL")

    def test_admission_needs_every_gate_for_this_freeze(self):
        G = self.G
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); (run / "gates").mkdir()
            config = {"_file_sha256": "abc"}
            for name in G.REQUIRED:
                (run / "gates" / f"{name}.json").write_text(json.dumps({"status": "PASS", "config_sha256": "abc"}))
            self.assertEqual(G.stage_admit(run, config)["status"], "PASS")
            (run / "gates" / "legacy.json").write_text(json.dumps({"status": "PASS", "config_sha256": "other"}))
            self.assertEqual(G.stage_admit(run, config)["status"], "FAIL")
            (run / "gates" / "legacy.json").write_text(json.dumps({"status": "PASS", "config_sha256": "abc"}))
            (run / "gates" / "tokens.json").unlink()
            out = G.stage_admit(run, config)
            self.assertEqual((out["status"], out["failed"]), ("FAIL", ["tokens"]))
        self.assertEqual(G.REQUIRED, ("tests", "tokens", "legacy", "gate", "devgate", "dev"))

    def test_previous_rows_come_from_the_runs_named_in_the_task(self):
        G = self.G; H = G.H
        self.assertEqual({m: p.name for m, p in G.PREVIOUS.items()},
                         {H.RECOMP: "mt_gqa_ours_order_20261120T052725Z", H.FULL_ASYNC: "mt_gqa_ours_order_20261120T052725Z",
                          H.ORIG_ASYNC: "mt_gqa_ours_order_20261120T052725Z", H.NEW_ASYNC: "mt_gqa_impress_code_20261117T185832Z"})

    @unittest.skipUnless(_transformers_major() == 4 and NEXT_SNAPSHOT.is_dir(), "needs the slow legacy tokenizer and the pinned snapshot")
    def test_gate1_passes_on_real_prompts_and_fails_when_the_recorded_suffix_is_not_the_legacy_path(self):
        G = self.G; H = G.H
        fast, slow, image = H.tokenizers()
        previous = []
        for m in G.PREVIOUS:
            for t, prompt in enumerate(_prompts(), 1):
                ids = fast(prompt).input_ids
                previous.append({"m": m, "d": "d0", "t": t, "i": "img", "prompt": prompt, "sfx": ids[ids.index(image) + 1:]})
        manifest = {"phase_dialogue_ids": {"main": ["d0"]}}
        out = G.gate1_tokens(previous, manifest)
        self.assertEqual(out["status"], "PASS")
        self.assertEqual(out["per_method"][H.RECOMP]["fixed_path_equal_original"], 3)
        self.assertEqual(out["per_method"][H.RECOMP]["legacy_path_equal_original"], 0)
        self.assertEqual(out["per_method"][H.RECOMP]["difference_is_one_token_after_image"], 3)
        previous[0] = dict(previous[0], sfx=previous[0]["sfx"][:-1])
        self.assertEqual(G.gate1_tokens(previous, manifest)["status"], "FAIL")
        self.assertEqual(G.gate1_tokens(previous[4:], manifest)["status"], "FAIL")      # a method's prompts are missing


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = _load("_test_mtgqa_main_v2_report", "224_report_mt_gqa_main_v2.py")

    def _records(self, flip=False):
        P = self.P
        records = []
        rng = np.random.default_rng(0)
        for i in range(6):
            rows = []
            for d in (f"i{i}d0", f"i{i}d1"):
                for m in P.METHODS:
                    for t in (1, 2, 3):
                        c = {"m": m, "d": d, "t": t, "score": float(rng.integers(0, 2)), "ttft": 100.0 + 10 * P.METHODS.index(m) + t, "e2e": 200.0,
                             "bytes": 0 if m == P.RECOMP else 1000, "reads": 0 if m == P.RECOMP else 33, "probe": 0, "k": 0, "v": 0, "packed": 0,
                             "blocked": None if m == P.RECOMP else 5.0, "prefill": 90.0, "hook": 20.0, "scatter": 3.0 if m == P.OURS else None,
                             "pid": 1, "at": 1.0, "tok": [1, 2 + (1 if flip and t == 3 else 0)], "pos": 0}
                        if m == P.IMP and t > 1:
                            c["imp"] = {"fb": [1, 0], "sim": [0.2, 0.4], "kc": [4, 2], "vc": [4, 2], "share": [0.9, 0.8], "raster": [4, 4], "only": [4, 3], "nc": 4,
                                        "blocked_probe": 1.0, "probe_read": 2.0, "ready": 1, "prefetched": 1}
                        rows.append(c)
            records.append({"image_id": f"i{i}", "rows": rows, "by": {(c["m"], c["d"], c["t"]): c for c in rows}, "dialogs": [f"i{i}d0", f"i{i}d1"],
                            "store": {}, "ours_table": {"idx": [1, 2], "thr": [1, 2], "counts": [2, 1], "v_token_num": 4, "newline_idx": []}})
        return records

    def test_impress_statistics_report_the_three_chunk_ratios(self):
        imp = self.P.impress_internal(self._records())
        self.assertAlmostEqual(imp["fallback_rate"], 0.5); self.assertAlmostEqual(imp["mean_jaccard"], 0.3)
        self.assertAlmostEqual(imp["kv_chunk_fraction"], 0.75)                       # all layers: (4/4 + 2/4) / 2
        self.assertAlmostEqual(imp["consensus_kv_chunk_fraction"], 0.5)              # layers that did not fall back
        self.assertAlmostEqual(imp["raster_chunk_fraction_computed"], 1.0)           # computed: had the store not been reordered
        self.assertAlmostEqual(imp["consensus_raster_chunk_fraction_computed"], 1.0)
        self.assertAlmostEqual(imp["prefix_share"], 0.85)

    def test_metacompress_rows(self):
        P = self.P
        arms = P.R.arm_table(self._records(), P.METHODS)
        meta = P.metacompress_table(arms, None)
        self.assertEqual(meta["metacompress_base_model"], {"Acc1": 63.83, "Acc2": 66.68, "Acc3": 67.94, "avg": 66.15})
        a = arms[0]
        self.assertAlmostEqual(meta["recomp_this_run"]["avg"], (a["em_T1"] + a["em_T2"] + a["em_T3"]) / 3)
        self.assertAlmostEqual(meta["difference_pp"]["Acc2"], a["em_T2"] - 66.68)

    def test_arm_table_has_every_turn_and_the_overall_em(self):
        P = self.P
        arms = P.R.arm_table(self._records(), P.METHODS)
        self.assertEqual([a["method"] for a in arms], ["ReComp", "FullLoad", "IMPRESS", "Ours"])
        for a in arms:
            self.assertAlmostEqual(a["em_all"], (a["em_T1"] + a["em_T2"] + a["em_T3"]) / 3)       # equal request counts per turn
            self.assertAlmostEqual(a["em_followup"], (a["em_T2"] + a["em_T3"]) / 2)

    def test_previous_comparison_counts_tokens_and_correctness(self):
        P = self.P
        records = self._records()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); (run / "reference").mkdir()
            old = []
            for r in records:
                for c in r["rows"]:
                    changed = c["m"] == P.OURS and c["t"] == 3 and c["d"].endswith("d0")
                    old.append({"m": c["m"], "d": c["d"], "t": c["t"], "i": r["image_id"], "tok": [9] if changed else c["tok"],
                                "score": 1.0 - c["score"] if changed else c["score"], "ttft": c["ttft"] + 1.0, "pred": "x"})
            with gzip.open(run / "reference" / "previous_rows.json.gz", "wt") as f:
                json.dump(old, f)
            tables = {r["image_id"]: dict(r["ours_table"]) for r in records}
            tables["i0"] = dict(tables["i0"], idx=[1, 3])
            with gzip.open(run / "reference" / "previous_ours_tables.json.gz", "wt") as f:
                json.dump(tables, f)
            (run / "reference" / "previous_sources.json").write_text(json.dumps({"sources": {}}))
            out = P.previous_comparison(run, records)
        by = {p["method_id"]: p for p in out["methods"]}
        self.assertEqual(by[P.RECOMP]["tokens_differ"], 0); self.assertEqual(by[P.RECOMP]["correctness_changed"], 0)
        self.assertEqual(by[P.OURS]["tokens_differ"], 6); self.assertEqual(by[P.OURS]["tokens_differ_by_turn"], {1: 0, 2: 0, 3: 6})
        self.assertEqual(by[P.OURS]["correctness_changed"], 6)
        self.assertEqual(by[P.OURS]["only_previous_correct"] + by[P.OURS]["only_this_run_correct"], 6)
        self.assertEqual(by[P.OURS]["requests"], 36)
        self.assertEqual((out["ours_token_lists"]["identical"], out["ours_token_lists"]["different"]), (5, 1))
        self.assertEqual(by[P.IMP]["previous_date"], "11/17"); self.assertEqual(by[P.FULL]["previous_date"], "11/20")

    def test_report_text_uses_only_the_four_method_names(self):
        P = self.P
        records = self._records()
        arms = P.R.arm_table(records, P.METHODS)
        text = "\n".join(P.main_tables_md(arms, P.R.paired_table(records, P.PAIRS), P.R.ratio_table(records, P.RATIOS), P.impress_internal(records),
                                          [], P.R.decomposition(records, P.METHODS), P.R.t1_identity(records, P.METHODS), None, P.metacompress_table(arms, None)))
        for forbidden in ("recompute", "fullload_async", "impress_official", "origorder", "packed_async", "row_order", "Selector"):
            self.assertNotIn(forbidden, text)
        for section in ("### 1. 본실험 표", "### 2. 짝지은 비교", "### 3. TTFT 비율", "### 4. 기존 결과와의 대조", "### 5. MetaCompress", "### 6. IMPRESS 통계",
                        "### 7. 재사용 턴", "### 8. T1 저장 비용"):
            self.assertIn(section, text)
        self.assertIn("## 9. 추가하거나 바꾼 코드", P.CODE_AND_LIMITS)
        self.assertEqual([c.split(" − ") for c in (p["comparison"] for p in P.R.paired_table(records, P.PAIRS))],
                         [["Ours", "IMPRESS"], ["Ours", "FullLoad"], ["Ours", "ReComp"], ["IMPRESS", "FullLoad"], ["FullLoad", "ReComp"]])
        self.assertEqual([r["ratio"] for r in P.R.ratio_table(records, P.RATIOS)],
                         ["Ours / FullLoad", "Ours / ReComp", "Ours / IMPRESS", "IMPRESS / FullLoad"])


class DriverTests(unittest.TestCase):
    def test_driver_is_detached_locked_and_asks_nothing(self):
        path = ROOT / "scripts/run_mt_gqa_main_v2.sh"
        self.assertEqual(subprocess.run(["bash", "-n", str(path)]).returncode, 0)
        text = path.read_text()
        for needle in ("setsid nohup", "flock -n 9", "driver.pid", "< /dev/null", "--resume", "phase main", "--stage admit", "NO_RETRY_"):
            self.assertIn(needle, text)
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        self.assertFalse(any(line.lstrip().startswith("read ") or " read -" in line for line in code))       # no prompt for input
        order = [text.index(s) for s in ("check tests", "check tokens", "phase legacy", "phase gate", "phase devgate", "phase dev;", "--stage admit", "phase main")]
        self.assertEqual(order, sorted(order))


if __name__ == "__main__":
    unittest.main()

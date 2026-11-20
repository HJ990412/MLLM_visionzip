#!/usr/bin/env python3
"""Gates of the MT-GQA run that compares Ours(재배열) with Ours(원래 순서)
(contract docs/mt_gqa_ours_order_contract.md, section 6).

  --stage cpu               gate 6: the whole CPU suite in this interpreter (and, when given, in a
                            second one); the new test file on its own
  --stage gate              on the committed rows of the ``gate`` phase (4 images): gate 1 (store
                            contents), gate 3 (original order: synchronous == asynchronous), gate 4
                            (ReComp / FullLoad / Ours(재배열) == the 2026-11-17 main run), gate 5
                            (original order executed twice)
  --stage qindep --phase P  gate 2 (GPU): for every image of phase P (``gate`` or ``devgate``) the
                            original-order store built from three different questions: selection and
                            layout must be identical; the K/V payload bytes are reported only
  --stage devgate           gates 3 and 5 on the 40 DEV images
  --stage dev               gates 1 and 4 on the 40 DEV images, and the DEV pilot's execution (gate 7)
  --stage admit             writes gates/admission.json (PASS only if every gate file is PASS); the
                            main phase refuses to start without it
  --stage posthoc           after the main phase (not a gate): gates 1 and 4 over all 398 images

Criteria are constants of this file, frozen with the run's source hash.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NEW_TEST = "test_packed_original_order.py"
BYTE_FIELDS = ["ssd_read_bytes", "packed_read_bytes"]
# metadata keys that describe the capturing request (ids, timings), not the store
PROVENANCE_KEYS = {"source_dialog_id", "source_turn_id", "source_method_id", "source_execution_id", "source_logical_request_id",
                   "turn1_capture", "packed_writer_timing_ms"}
PAYLOAD_DIGEST_KEY = "packed_sha256"      # digest of the K/V payload bytes: reported, not part of gate 2


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("188_validate_mt_gqa_async.py", "_mtgqa_async_gates")      # write / committed_rows / compare / run_suite / baselines
write, committed_rows, compare = G.write, G.committed_rows, G.compare


def load_harness():
    return _load("202_eval_mt_gqa_ours_order.py", "_mtgqa_ours_order_harness")


# ------------------------------------------------------------------ gate 6
def stage_cpu(run, extra_python):
    suites = [G.run_suite(sys.executable)]
    if extra_python:
        suites.append(G.run_suite(extra_python))
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", NEW_TEST], capture_output=True, text=True, cwd=ROOT)
    ran = re.search(r"^Ran (\d+) tests", p.stdout + p.stderr, re.M)
    out = {"gate": "gate6_existing_tests", "suites": suites,
           "new_unit_tests": {"file": f"tests/{NEW_TEST}", "ran": int(ran.group(1)) if ran else None, "ok": p.returncode == 0},
           "criterion": "the whole suite (python -m unittest discover -s tests) passes in both environments, except the environment-specific "
                        "failures that exist without this work (transformers 4.57.6: the 3 GLM-4.6V tests; 5.13.1: 2 vision-capture tests); "
                        "the new test file passes",
           "status": "PASS" if all(r["status"] == "PASS" for r in suites) and p.returncode == 0 and ran else "FAIL"}
    write(run / "gates" / "gate6_tests.json", out)
    print(json.dumps({"gate6": out["status"], "suites": [{k: r[k] for k in ("transformers", "ran", "failed", "new_failures")} for r in suites],
                      "new_unit_tests": out["new_unit_tests"]}, indent=1))
    return out["status"]


# ------------------------------------------------------------- shared checks
def phase_keys(run, phase):
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    keys = {(d, t) for d in manifest["phase_dialogue_ids"][phase] for t in (1, 2, 3)}
    validation = run / phase / "validation.json"
    complete = validation.exists() and json.loads(validation.read_text())["status"] == "PASS"
    return manifest, config, keys, config["phases"][phase]["methods"], complete


def attempts_of(run, phase):
    out = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        out[cp.parent.name] = cp.parent / json.loads(cp.read_text())["attempt_id"]
    return out


def execution(run, phase, rows, manifest, complete):
    return {"gate": f"{phase}_phase_execution", "requests": len(rows), "expected": manifest["expected_requests"][phase],
            "images": len({r["image_id"] for r in rows.values()}), "phase_validation_pass": complete,
            "criterion": "every image committed after its independent audit passed",
            "status": "PASS" if complete and len(rows) == manifest["expected_requests"][phase] else "FAIL"}


def gate1_store_contents(run, phase, manifest):
    """Per image: the original-order store against the importance store (read back from disk by the harness)."""
    images = []
    expected = manifest["phase_images"][phase]
    attempts = attempts_of(run, phase)
    for image in expected:
        folder = attempts.get(image)
        p = folder / "store_equivalence.json" if folder else None
        if p is None or not p.exists():
            images.append({"image_id": image, "status": "MISSING"}); continue
        eq = json.loads(p.read_text())
        images.append({"image_id": image, **{k: eq[k] for k in (
            "status", "layers", "layer_sets_equal", "byte_table_equal", "payload_bytes_equal", "payload_bytes", "kv_rows_compared",
            "kv_rows_different", "separator_block_equal", "separator_bytes", "system_kv_equal", "original_metadata_has_no_importance_permutation")}})
    return {"criterion": "for every image: layer l's original indices of the original-order store == the first k_l entries of the importance "
                         "store's order (as sets); every stored token's K and V rows bit-identical between the two files; separator block "
                         "bit-identical; payload bytes equal",
            "images": images, "image_count": len(images), "pass_count": sum(x["status"] == "PASS" for x in images),
            "kv_rows_compared": sum(x.get("kv_rows_compared", 0) for x in images), "kv_rows_different": sum(x.get("kv_rows_different", 0) for x in images),
            "status": "PASS" if images and all(x["status"] == "PASS" for x in images) else "FAIL"}


def continuity(H, rows, keys, methods):
    """Generated tokens of this run against the 2026-11-17 main run's record of the same (arm, dialogue, turn)."""
    images = {r["image_id"] for r in rows.values()}
    old_rows = committed_rows(H.PRIOR_MAIN_RUN, "main", set(methods), images)
    out = []
    for m in methods:
        mism = []; n = 0; prompt_equal = 0
        for dialog, turn in sorted(keys):
            new, old = rows.get((m, dialog, turn)), old_rows.get((m, dialog, turn))
            if new is None or old is None:
                mism.append({"dialog_id": dialog, "turn_id": turn, "reason": "missing in " + ("this run" if new is None else "the 2026-11-17 run")}); continue
            n += 1; prompt_equal += new["prompt"] == old["prompt"]
            if new["result"]["generated_token_ids"] != old["result"]["generated_token_ids"]:
                mism.append({"dialog_id": dialog, "turn_id": turn, "image_id": new["image_id"], "new": new["prediction"], "previous": old["prediction"],
                             "new_tokens": new["result"]["generated_token_ids"], "previous_tokens": old["result"]["generated_token_ids"],
                             "prompt_equal": new["prompt"] == old["prompt"]})
        out.append({"method": m, "name": H.ARMS[m]["display"], "requests": n, "expected": len(keys), "prompt_equal": prompt_equal,
                    "mismatch_count": len(mism), "mismatches": mism[:50], "status": "PASS" if n == len(keys) and not mism else "FAIL"})
    return {"criterion": "ReComp, FullLoad, Ours(재배열): generated tokens identical to the 2026-11-17 main run's record of the same request "
                         "(same arm, dialogue, turn)",
            "reference": str((H.PRIOR_MAIN_RUN / "main").relative_to(ROOT)), "methods": out,
            "status": "PASS" if out and all(x["status"] == "PASS" for x in out) else "FAIL"}


def turn1_identity(rows, keys, methods):
    dialogs = sorted({d for d, t in keys if t == 1}); bad = []
    for d in dialogs:
        got = {m: rows[(m, d, 1)]["result"]["generated_token_ids"] for m in methods if (m, d, 1) in rows}
        if len(got) != len(methods) or len({tuple(v) for v in got.values()}) != 1:
            bad.append({"dialog_id": d, "tokens": got})
    return {"criterion": "none (reported only): turn-1 generated tokens identical across all arms of the phase, per dialogue",
            "dialogues": len(dialogs), "arms": list(methods), "mismatch_count": len(bad), "mismatches": bad[:20], "status": "REPORTED"}


def two_ours_difference(H, rows, keys):
    """Reported, never a gate: requests whose tokens differ between the two packed arms."""
    out = {}
    for turn in (1, 2, 3):
        n = diff = only_a = only_b = same_prompt_diff = 0
        for dialog, t in sorted(keys):
            if t != turn:
                continue
            a, b = rows.get((H.OURS_ASYNC, dialog, t)), rows.get((H.ORIG_ASYNC, dialog, t))
            if a is None or b is None:
                continue
            n += 1
            if a["result"]["generated_token_ids"] != b["result"]["generated_token_ids"]:
                diff += 1; same_prompt_diff += a["prompt"] == b["prompt"]
            only_a += a["score"] > b["score"]; only_b += b["score"] > a["score"]
        out[f"t{turn}"] = {"requests": n, "tokens_differ": diff, "tokens_differ_with_equal_prompt": same_prompt_diff,
                           "only_reorder_correct": only_a, "only_original_correct": only_b}
    return {"criterion": "none (reported only): a different key order changes the BF16 summation order, so boundary cases may flip",
            "by_turn": out, "status": "REPORTED"}


# ------------------------------------------------------ gates 1, 3, 4, 5 (4 images)
def stage_gate(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "gate")
    rows = committed_rows(run, "gate")
    ex = execution(run, "gate", rows, manifest, complete)
    g1 = {"gate": "gate1_store_contents_gate", **gate1_store_contents(run, "gate", manifest)}
    g3 = {"gate": "gate3_original_order_sync_equals_async_gate",
          "criterion": "Ours(원래 순서): generated tokens and bytes read identical between the synchronous and the asynchronous arm, every request",
          **compare(rows, H.ORIG_ASYNC, H.ORIG, ["generated_token_ids"] + BYTE_FIELDS, keys)}
    g4 = {"gate": "gate4_continuity_gate", **continuity(H, rows, keys, ["recompute", "fullload_async", H.OURS_ASYNC])}
    g5 = {"gate": "gate5_original_order_repeat_gate",
          "criterion": "Ours(원래 순서) executed twice on the same requests: generated tokens identical, every request",
          **compare(rows, H.ORIG_REPEAT, H.ORIG_ASYNC, ["generated_token_ids"], keys)}
    t1 = {"gate": "reported_turn1_identity_gate", **turn1_identity(rows, keys, methods)}
    d2 = {"gate": "reported_two_ours_difference_gate", **two_ours_difference(H, rows, keys)}
    for name, obj in (("gate_execution", ex), ("gate1_store_contents_gate", g1), ("gate3_sync_equals_async_gate", g3),
                      ("gate4_continuity_gate", g4), ("gate5_repeat_gate", g5), ("reported_turn1_identity_gate", t1),
                      ("reported_two_ours_difference_gate", d2)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": ex["status"], "gate1": g1["status"], "gate3": g3["status"], "gate4": g4["status"], "gate5": g5["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"gate1": {k: g1[k] for k in ("image_count", "pass_count", "kv_rows_compared", "kv_rows_different")},
                      "gate3": {k: g3[k] for k in ("requests", "equal", "mismatch_count")},
                      "gate4": [{k: m[k] for k in ("method", "requests", "prompt_equal", "mismatch_count")} for m in g4["methods"]],
                      "gate5": {k: g5[k] for k in ("requests", "equal", "mismatch_count")},
                      "turn1": {k: t1[k] for k in ("dialogues", "mismatch_count")}, "two_ours": d2["by_turn"]}, indent=1))
    return "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"


# ------------------------------------------------------------------- gate 2
def stage_qindep(run, phase):
    """Question independence of the original-order store (GPU).

    For every image of the phase, three NEW pixel requests with three different
    questions each build an original-order store through the run's own
    persistence path.

    Criterion (what is stored and where): the per-layer original-index tables,
    the row counts, the byte table, the file set and every metadata field that
    is not a record of the capturing request are identical across the three
    stores, and the index tables equal those of the store the phase committed.

    Reported, not judged (user decision of 2026-11-20, contract section 6): the
    K/V payload bytes.  The visual-prefix K/V captured in turn 1 depends on the
    token LENGTH of the capturing prompt, so prompts of different lengths give
    different payload bytes in every store type.  Per image this stage records
    which pairs of questions have equal payloads, the fraction of FP16 elements
    that differ, and -- from the same three captures -- the payload digests of
    the importance-order store, which follow the same pattern.
    """
    import torch
    from mmimpress.packed_store import PACKED_NAME, layer_original_indices, packed_file_sha256
    H = load_harness()
    assert phase in ("gate", "devgate")
    H.verify_freeze(run)
    manifest, config, keys, methods, complete = phase_keys(run, phase)
    assert complete, f"{phase} phase is not complete"
    runner, servers = H.load_model(config)
    tag = "gate" if phase == "gate" else "dev"
    images = []
    try:
        ids = set(manifest["phase_dialogue_ids"][phase])
        for image, folder in attempts_of(run, phase).items():
            committed = json.loads((folder / f"{H.ORIG_KIND}_meta.json").read_text())
            ds = [d for d in manifest["dialogues"] if d["image_id"] == image and d["dialog_id"] in ids]
            questions = []
            for d in ds:
                for turn in d["turns"]:
                    q = turn["question"].strip()
                    if q not in questions:
                        questions.append(q)
            assert len(questions) >= 3, (image, questions)
            questions = questions[:3]
            path = ROOT / ds[0]["image_path"]
            scratch = run / "scratch" / "qindep" / phase / image
            assert scratch.resolve().is_relative_to((run / "scratch").resolve()) and not scratch.exists()
            scratch.mkdir(parents=True)
            builds = []
            for i, q in enumerate(questions):
                prompt = "USER: <image>\n" + "\n".join([f"Current question Q1: {q}", f"{H.SHORT_ANSWER} ASSISTANT:"])
                result, diag = H.normal_request(runner, servers["base"], path, lambda: (prompt, "", []), "image_only")
                source = {"dialog_id": ds[0]["dialog_id"], "method_id": H.ORIG_ASYNC, "physical_execution_id": f"qindep-{image}-{i}",
                          "logical_request_id": f"qindep:{run.name}:{phase}:{image}:q{i}"}
                # the same capture written in both row orders: the original-order store (judged) and the importance-order store (reported)
                H.persist(runner, result, diag, scratch / f"q{i}", image, H.ORIG_KIND, source, config["layer_budgets"])
                H.persist(runner, result, diag, scratch / f"q{i}_reorder", image, H.PACKED_KIND, source, config["layer_budgets"])
                suffix = H.QA._suffix_from_tokenized(runner, runner.processor.tokenizer(prompt, return_tensors="pt"))
                result.pop("captured_past_key_values", None)
                meta = json.loads((scratch / f"q{i}" / "meta.json").read_text())
                builds.append({"question": q, "suffix_tokens": int(suffix.numel()), "meta": meta,
                               "packed_sha256": packed_file_sha256(scratch / f"q{i}" / PACKED_NAME),
                               "reorder_packed_sha256": packed_file_sha256(scratch / f"q{i}_reorder" / PACKED_NAME),
                               "files": sorted(p.name for p in (scratch / f"q{i}").iterdir()),
                               "payload": np.fromfile(scratch / f"q{i}" / PACKED_NAME, dtype=np.float16),
                               "sys": torch.load(scratch / f"q{i}" / "sys_kv.pt", weights_only=True)})
                del result, diag
            ref = builds[0]
            # ---- judged: what is stored and where
            tables_equal = all(layer_original_indices(b["meta"]) == layer_original_indices(ref["meta"])
                               and b["meta"]["stored_original_indices"] == ref["meta"]["stored_original_indices"]
                               and b["meta"]["stored_min_keep_count"] == ref["meta"]["stored_min_keep_count"] for b in builds)
            counts_equal = all(b["meta"]["layer_keep_counts"] == ref["meta"]["layer_keep_counts"] for b in builds)
            byte_table_equal = all(b["meta"][k] == ref["meta"][k] for b in builds
                                   for k in ("packed_layout", "packed_sep_offset", "packed_sep_bytes", "packed_total_bytes", "packed_row_bytes", "newline_idx"))
            differing = sorted({k for b in builds for k in set(b["meta"]) | set(ref["meta"]) if b["meta"].get(k) != ref["meta"].get(k)})
            meta_equal = set(differing) <= PROVENANCE_KEYS | {PAYLOAD_DIGEST_KEY}
            files_equal = all(b["files"] == ref["files"] for b in builds) and "visionzip_layout.pt" not in ref["files"]
            flags = all(b["meta"].get("layout_uses_dataset_question") is False and b["meta"].get("llm_used_for_layout_scoring") is False
                        and b["meta"].get("calibration_questions") == 0 for b in builds)
            committed_tables_equal = (committed["stored_original_indices"] == ref["meta"]["stored_original_indices"]
                                      and committed["stored_min_keep_count"] == ref["meta"]["stored_min_keep_count"]
                                      and committed["layer_keep_counts"] == ref["meta"]["layer_keep_counts"]
                                      and committed["packed_layout"] == ref["meta"]["packed_layout"])
            ok = tables_equal and counts_equal and byte_table_equal and meta_equal and files_equal and flags and committed_tables_equal
            # ---- reported: the K/V payload bytes
            pairs = []
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = builds[i], builds[j]
                    neq = a["payload"] != b["payload"]
                    n_diff = int(neq.sum())
                    pairs.append({"questions": [i, j], "suffix_tokens": [a["suffix_tokens"], b["suffix_tokens"]],
                                  "equal_prompt_length": a["suffix_tokens"] == b["suffix_tokens"],
                                  "payload_equal": a["packed_sha256"] == b["packed_sha256"],
                                  "reorder_payload_equal": a["reorder_packed_sha256"] == b["reorder_packed_sha256"],
                                  "fp16_elements_differ": n_diff, "fp16_elements": int(a["payload"].size),
                                  "fraction_differ": n_diff / int(a["payload"].size),
                                  "max_abs_diff": float(np.abs(a["payload"][neq].astype(np.float32) - b["payload"][neq].astype(np.float32)).max()) if n_diff else 0.0})
            images.append({"image_id": image, "questions": questions, "suffix_tokens": [b["suffix_tokens"] for b in builds],
                           "distinct_suffix_lengths": len({b["suffix_tokens"] for b in builds}),
                           "index_tables_equal": tables_equal, "row_counts_equal": counts_equal, "byte_table_equal": byte_table_equal,
                           "metadata_equal_outside_request_records_and_payload_digest": meta_equal, "metadata_keys_that_differ": differing,
                           "file_sets_equal_no_importance_artifact": files_equal, "no_question_llm_calibration_flags": flags,
                           "index_tables_equal_to_committed_store": committed_tables_equal,
                           "reported_payload": {
                               "all_three_payloads_equal": len({b["packed_sha256"] for b in builds}) == 1,
                               "first_question_payload_equals_committed_store": committed["packed_sha256"] == ref["packed_sha256"],
                               "system_kv_equal": all(torch.equal(b["sys"][n], ref["sys"][n]) for b in builds for n in ("k", "v")),
                               "packed_sha256": [b["packed_sha256"] for b in builds], "committed_packed_sha256": committed["packed_sha256"],
                               "reorder_packed_sha256": [b["reorder_packed_sha256"] for b in builds], "pairs": pairs},
                           "payload_bytes": int(ref["meta"]["packed_total_bytes"]), "status": "PASS" if ok else "FAIL"})
            del builds, ref
            shutil.rmtree(scratch)
            print(json.dumps({"image_id": image, "status": images[-1]["status"], "suffix_tokens": images[-1]["suffix_tokens"],
                              "index_tables_equal": tables_equal, "metadata_keys_that_differ": differing,
                              "payload_pairs": [(x["suffix_tokens"], x["payload_equal"], x["reorder_payload_equal"], round(x["fraction_differ"], 4)) for x in pairs]}), flush=True)
    finally:
        for server in servers.values():
            server.close()
    expected = manifest["phase_images"][phase]
    all_pairs = [x for im in images for x in im["reported_payload"]["pairs"]]
    same_len = [x for x in all_pairs if x["equal_prompt_length"]]; other = [x for x in all_pairs if not x["equal_prompt_length"]]
    out = {"gate": f"gate2_question_independent_{tag}",
           "criterion": "for every image, the original-order stores built from three different questions hold the same selection and layout: "
                        "per-layer original-index tables, row counts, byte table, file set, and every metadata field except the record of the "
                        "capturing request and the payload digest; and the index tables equal those of the phase's committed store. "
                        "The inputs of the row plan are the VisionZip order and the frozen layer counts only (tests/test_packed_original_order.py)",
           "reported_not_judged": "the K/V payload bytes: the visual-prefix K/V captured in turn 1 depends on the token length of the capturing "
                                  "prompt (every store type); see payload_summary and each image's reported_payload",
           "inputs_of_the_store": "captured Turn-1 visual-prefix K/V, CLIP penultimate-layer CLS-to-patch attention (VisionZip score), "
                                  "separator positions, the frozen PYR-beta-raw layer budget; no question text, no decoder attention, no "
                                  "calibration score",
           "payload_summary": {
               "question_pairs": len(all_pairs),
               "equal_prompt_length_pairs": len(same_len), "equal_prompt_length_pairs_with_equal_payload": sum(x["payload_equal"] for x in same_len),
               "different_prompt_length_pairs": len(other), "different_prompt_length_pairs_with_equal_payload": sum(x["payload_equal"] for x in other),
               "mean_fraction_of_fp16_elements_that_differ_when_lengths_differ": float(np.mean([x["fraction_differ"] for x in other])) if other else None,
               "max_abs_diff_when_lengths_differ": max((x["max_abs_diff"] for x in other), default=None),
               "pairs_where_importance_order_store_agrees_with_original_order_store_on_equal_or_different":
                   sum(x["payload_equal"] == x["reorder_payload_equal"] for x in all_pairs),
               "images_with_all_three_payloads_equal": sum(im["reported_payload"]["all_three_payloads_equal"] for im in images),
               "images_whose_first_question_payload_equals_committed_store": sum(im["reported_payload"]["first_question_payload_equals_committed_store"] for im in images),
               "images_with_equal_system_kv": sum(im["reported_payload"]["system_kv_equal"] for im in images)},
           "images": images, "image_count": len(images), "pass_count": sum(x["status"] == "PASS" for x in images),
           "status": "PASS" if images and len(images) == len(expected) and all(x["status"] == "PASS" for x in images) else "FAIL"}
    write(run / "gates" / f"gate2_question_independent_{tag}.json", out)
    print(json.dumps({"gate2": out["status"], "images": len(images), "pass": out["pass_count"], "payload": out["payload_summary"]}))
    return out["status"]


# ------------------------------------------------------ gates 3, 5 (DEV 40)
def stage_devgate(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "devgate")
    rows = committed_rows(run, "devgate")
    ex = execution(run, "devgate", rows, manifest, complete)
    g3 = {"gate": "gate3_original_order_sync_equals_async_dev",
          "criterion": "Ours(원래 순서): generated tokens and bytes read identical between the synchronous and the asynchronous arm, every request",
          **compare(rows, H.ORIG_ASYNC, H.ORIG, ["generated_token_ids"] + BYTE_FIELDS, keys)}
    g5 = {"gate": "gate5_original_order_repeat_dev",
          "criterion": "Ours(원래 순서) executed twice on the same requests: generated tokens identical, every request",
          **compare(rows, H.ORIG_REPEAT, H.ORIG_ASYNC, ["generated_token_ids"], keys)}
    for name, obj in (("devgate_execution", ex), ("gate3_sync_equals_async_dev", g3), ("gate5_repeat_dev", g5)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": ex["status"], "gate3": g3["status"], "gate5": g5["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"gate3": {k: g3[k] for k in ("requests", "equal", "mismatch_count")},
                      "gate5": {k: g5[k] for k in ("requests", "equal", "mismatch_count")}}, indent=1))
    return "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"


# ------------------------------------------------------ gates 1, 4, 7 (DEV 40)
def stage_dev(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "dev")
    rows = committed_rows(run, "dev")
    ex = execution(run, "dev", rows, manifest, complete)
    g1 = {"gate": "gate1_store_contents_dev", **gate1_store_contents(run, "dev", manifest)}
    g4 = {"gate": "gate4_continuity_dev", **continuity(H, rows, keys, ["recompute", "fullload_async", H.OURS_ASYNC])}
    t1 = {"gate": "reported_turn1_identity_dev", **turn1_identity(rows, keys, methods)}
    d2 = {"gate": "reported_two_ours_difference_dev", **two_ours_difference(H, rows, keys)}
    for name, obj in (("dev_execution", ex), ("gate1_store_contents_dev", g1), ("gate4_continuity_dev", g4),
                      ("reported_turn1_identity_dev", t1), ("reported_two_ours_difference_dev", d2)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": ex["status"], "gate1": g1["status"], "gate4": g4["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"gate1": {k: g1[k] for k in ("image_count", "pass_count", "kv_rows_compared", "kv_rows_different")},
                      "gate4": [{k: m[k] for k in ("method", "requests", "prompt_equal", "mismatch_count")} for m in g4["methods"]],
                      "turn1": {k: t1[k] for k in ("dialogues", "mismatch_count")}, "two_ours": d2["by_turn"]}, indent=1))
    return "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"


# -------------------------------------------------------------------- admit
REQUIRED = ("gate_execution", "gate1_store_contents_gate", "gate2_question_independent_gate", "gate3_sync_equals_async_gate",
            "gate4_continuity_gate", "gate5_repeat_gate", "gate6_tests",
            "devgate_execution", "gate2_question_independent_dev", "gate3_sync_equals_async_dev", "gate5_repeat_dev",
            "dev_execution", "gate1_store_contents_dev", "gate4_continuity_dev")


def stage_admit(run):
    H = load_harness()
    statuses = {}
    for name in REQUIRED:
        p = run / "gates" / f"{name}.json"
        statuses[name] = json.loads(p.read_text())["status"] if p.exists() else "MISSING"
    ok = all(v == "PASS" for v in statuses.values())
    out = {"status": "PASS" if ok else "FAIL", "gates": statuses, "config_sha256": H.sha(run / "config.json"),
           "decided_unix": time.time(), "rule": "main runs only if gates 1-6 are PASS on the 4 gate images and on the 40 DEV images and the DEV pilot (gate 7) completed"}
    write(run / "gates" / "admission.json", out)
    print(json.dumps(out, indent=1))
    return out["status"]


def stage_posthoc(run):
    """After the main phase; decides only whether the comparison with the stored IMPRESS rows is reported."""
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "main")
    rows = committed_rows(run, "main")
    g4 = {"gate": "posthoc_continuity_main", **continuity(H, rows, keys, ["recompute", "fullload_async", H.OURS_ASYNC]), "phase_complete": complete}
    g1 = {"gate": "posthoc_store_contents_main", **gate1_store_contents(run, "main", manifest)}
    t1 = {"gate": "posthoc_turn1_identity_main", **turn1_identity(rows, keys, methods)}
    for x in g1["images"]:      # 398 entries: keep the per-image list short
        for k in ("layers", "byte_table_equal", "payload_bytes_equal", "separator_bytes", "system_kv_equal"):
            x.pop(k, None)
    write(run / "gates" / "posthoc_continuity_main.json", g4)
    write(run / "gates" / "posthoc_store_contents_main.json", g1)
    write(run / "gates" / "posthoc_turn1_identity_main.json", t1)
    print(json.dumps({"complete": complete, "continuity": [{k: m[k] for k in ("method", "requests", "mismatch_count")} for m in g4["methods"]],
                      "store_contents": {k: g1[k] for k in ("image_count", "pass_count", "kv_rows_different")},
                      "turn1": {k: t1[k] for k in ("dialogues", "mismatch_count")}}, indent=1))
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=("cpu", "gate", "qindep", "devgate", "dev", "admit", "posthoc"), required=True)
    ap.add_argument("--phase", choices=("gate", "devgate"), default="gate", help="qindep: which phase's images")
    ap.add_argument("--extra-python", default=None, help="second interpreter for the CPU suite (cpu stage)")
    args = ap.parse_args()
    run = args.run.resolve()
    status = {"cpu": lambda: stage_cpu(run, args.extra_python), "gate": lambda: stage_gate(run), "qindep": lambda: stage_qindep(run, args.phase),
              "devgate": lambda: stage_devgate(run), "dev": lambda: stage_dev(run), "admit": lambda: stage_admit(run),
              "posthoc": lambda: stage_posthoc(run)}[args.stage]()
    if status != "PASS":
        raise SystemExit(f"{args.stage}: {status}")


if __name__ == "__main__":
    main()

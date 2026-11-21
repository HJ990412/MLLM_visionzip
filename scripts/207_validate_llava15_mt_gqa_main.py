#!/usr/bin/env python3
"""Gates of the LLaVA-1.5 x MT-GQA main experiment
(contract docs/llava15_mt_gqa_main_contract.md, section 7).

  --stage tests             gate 7: the whole unit-test suite in the run environment and, as a
                            cross-check, in the transformers 5.13.1 environment; the new test
                            file on its own
  --stage gate              from the committed ``gate`` phase (4 images): gate 1 (Ours store
                            contents), gate 3 (Ours: synchronous == asynchronous), gate 4 (Ours
                            executed twice), gate 5 (IMPRESS at 100 % == the FullLoad loader on
                            the same reordered store; IMPRESS synchronous == asynchronous),
                            gate 6 (ReComp == the 2026-11-12 run)
  --stage qindep --phase P  gate 2 (GPU): for every image of phase P (``gate`` or ``devgate``)
                            the Ours store built from three different questions
  --stage devgate           gates 3, 4, 5 on the 40 DEV images
  --stage dev               gates 1, 6 on the 40 DEV images and the execution of the DEV pilot
                            (gate 8)
  --stage admit             writes gates/admission.json (PASS only if every gate file is PASS for
                            this freeze); the main phase refuses to start without it
  --stage posthoc           after the main phase (reported, not a gate): gate 6's comparison over
                            all 398 images and the turn-1 identity

Criteria are constants of this file, frozen with the run's source hash.  A FAIL is never
re-judged: a gate file that already exists for this freeze is kept (``.superseded_`` copies
only appear when the stage is run again by hand).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llava15_support as S  # noqa: E402

H = S.load_script("_l15_mtgqa_main_eval", "206_eval_llava15_mt_gqa_main.py")
G = S.load_script("_mtgqa_async_gates", "188_validate_mt_gqa_async.py")          # run_suite, BASELINE_FAILURES, committed_rows (shared helper)
sha, atomic_json, read_rows = H.sha, H.atomic_json, H.read_rows
PY_CROSS = str(ROOT / ".venv-glm46v/bin/python")
NEW_TEST = "test_llava15_original_order.py"
BYTE_FIELDS = ("ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes", "packed_read_bytes")
# metadata keys that describe the capturing request (ids, timings), not the store
PROVENANCE_KEYS = {"source_dialog_id", "source_turn_id", "source_method_id", "source_execution_id", "source_logical_request_id",
                   "turn1_capture", "packed_writer_timing_ms"}
PAYLOAD_DIGEST_KEY = "packed_sha256"      # digest of the K/V payload bytes: reported, not part of gate 2
REQUIRED = ("tests", "gate", "qindep_gate", "devgate", "qindep_devgate", "dev")


def frozen(run):
    config = json.loads((run / "config.json").read_text())
    config["_file_sha256"] = sha(run / "config.json")
    return config


def receipt(run, name, doc, config):
    out = run / "gates" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    doc = dict(doc, config_sha256=config["_file_sha256"], written_unix=time.time())
    if out.exists():      # a superseded receipt is kept, never overwritten
        os.replace(out, out.with_name(f"{name}.superseded_{time.strftime('%Y%m%dT%H%M%S')}.json"))
    atomic_json(out, doc)
    print(json.dumps({"stage": name, "status": doc["status"]}, ensure_ascii=False), flush=True)
    return doc


# ---------------------------------------------------------------- committed io
def committed_rows(run, phase):
    """{(method, dialog_id, turn): row} of every committed image of a phase."""
    rows = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text())
        got = read_rows(cp.parent / doc["attempt_id"] / "raw.jsonl")
        assert len(got) == doc["requests"]
        for r in got:
            rows[(r["method_id"], r["dialog_id"], r["turn_id"])] = r
    return rows


def attempts_of(run, phase):
    out = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        out[cp.parent.name] = cp.parent / json.loads(cp.read_text())["attempt_id"]
    return out


def tokens(row):
    return row["result"]["generated_token_ids"]


def phase_state(run, phase, config):
    manifest = json.loads((run / "manifest.json").read_text())
    keys = sorted((d, t) for d in manifest["phase_dialogue_ids"][phase] for t in (1, 2, 3))
    path = run / phase / "validation.json"
    v = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
    ok = (v.get("status") == "PASS" and v.get("requests") == manifest["expected_requests"][phase]
          and v.get("config_sha256") == config["_file_sha256"] and v.get("independent_image_audit") == "PASS")
    execution = dict(v, status="PASS" if ok else "FAIL", expected=manifest["expected_requests"][phase],
                     criterion="every image of the phase committed after its independent audit passed, for this freeze")
    return manifest, keys, execution


def compare(rows, a, b, keys, fields=()):
    """Per-request equality of the generated tokens (and ``fields``) between arms ``a`` and ``b``."""
    out = {"arms": [a, b], "requests": 0, "tokens_differ": 0, "tokens_differ_same_prompt": 0, "prompt_differs": 0,
           "fields_differ": {f: 0 for f in fields}, "missing": 0, "examples": []}
    for dialog, turn in keys:
        ra, rb = rows.get((a, dialog, turn)), rows.get((b, dialog, turn))
        if ra is None or rb is None:
            out["missing"] += 1
            continue
        out["requests"] += 1
        same_prompt = ra["prompt"] == rb["prompt"]
        out["prompt_differs"] += not same_prompt
        bad = [f for f in fields if ra[f] != rb[f]]
        for f in bad:
            out["fields_differ"][f] += 1
        if tokens(ra) != tokens(rb):
            out["tokens_differ"] += 1; out["tokens_differ_same_prompt"] += same_prompt
            bad.append("generated_token_ids")
        if bad and len(out["examples"]) < 30:
            out["examples"].append({"image_id": ra["image_id"], "dialog_id": dialog, "turn_id": turn, "fields": bad, "same_prompt": same_prompt,
                                    a: tokens(ra), b: tokens(rb), "predictions": [ra["prediction"], rb["prediction"]]})
    out["status"] = "PASS" if (out["requests"] == len(keys) and not out["missing"] and not out["tokens_differ"]
                               and not any(out["fields_differ"].values())) else "FAIL"
    return out


def gate1_store_contents(run, phase, manifest):
    """Per image: the Ours store against the reordered store of the same image (compared on disk by the harness)."""
    images = []
    attempts = attempts_of(run, phase)
    for image in manifest["phase_images"][phase]:
        folder = attempts.get(image)
        p = folder / "store_equivalence.json" if folder else None
        if p is None or not p.exists():
            images.append({"image_id": image, "status": "MISSING"})
            continue
        eq = json.loads(p.read_text())
        images.append({"image_id": image, **{k: eq[k] for k in (
            "status", "layers", "layer_sets_equal", "byte_table_equal", "payload_bytes_equal", "payload_bytes", "original_payload_bytes",
            "kv_rows_compared", "kv_rows_different", "system_kv_equal", "original_metadata_has_no_importance_permutation")}})
    return {"criterion": "for every image: layer l's original indices of the Ours store == the first k_l entries of the reordered store's order "
                         "(as sets, ties included); every stored token's K and V rows bit-identical between the two files; payload bytes equal",
            "images": images, "image_count": len(images), "pass_count": sum(x["status"] == "PASS" for x in images),
            "kv_rows_compared": sum(x.get("kv_rows_compared", 0) for x in images), "kv_rows_different": sum(x.get("kv_rows_different", 0) for x in images),
            "status": "PASS" if images and all(x["status"] == "PASS" for x in images) else "FAIL"}


def gate6_recomp(rows, keys):
    """ReComp tokens of this run against the 2026-11-12 run's record of the same (dialogue, turn)."""
    images = {r["image_id"] for r in rows.values()}
    old = G.committed_rows(H.RECOMP_REF_RUN, "main", {"recompute"}, images)
    out = {"criterion": "ReComp: generated tokens identical to the 2026-11-12 run's record of the same request (same dialogue, same turn)",
           "reference": str((H.RECOMP_REF_RUN / "main").relative_to(ROOT)), "requests": 0, "prompt_equal": 0, "missing": 0, "mismatch_count": 0, "mismatches": []}
    for dialog, turn in keys:
        new, ref = rows.get(("recompute", dialog, turn)), old.get(("recompute", dialog, turn))
        if new is None or ref is None:
            out["missing"] += 1
            continue
        out["requests"] += 1; out["prompt_equal"] += new["prompt"] == ref["prompt"]
        if tokens(new) != tokens(ref):
            out["mismatch_count"] += 1
            if len(out["mismatches"]) < 50:
                out["mismatches"].append({"image_id": new["image_id"], "dialog_id": dialog, "turn_id": turn, "prompt_equal": new["prompt"] == ref["prompt"],
                                          "new": new["prediction"], "previous": ref["prediction"], "new_tokens": tokens(new), "previous_tokens": tokens(ref)})
    out["status"] = "PASS" if out["requests"] == len(keys) and not out["missing"] and not out["mismatch_count"] else "FAIL"
    return out


def turn1_identity(rows, keys, methods):
    dialogs = sorted({d for d, t in keys if t == 1}); bad = []
    for d in dialogs:
        got = {m: tokens(rows[(m, d, 1)]) for m in methods if (m, d, 1) in rows}
        if len(got) != len(methods) or len({tuple(v) for v in got.values()}) != 1:
            bad.append({"dialog_id": d, "tokens": got})
    return {"criterion": "none (reported only): turn-1 generated tokens identical across all arms of the phase, per dialogue",
            "dialogues": len(dialogs), "arms": list(methods), "mismatch_count": len(bad), "mismatches": bad[:20], "status": "REPORTED"}


def sync_async_timing(rows, keys, pairs):
    """Reference only (not a criterion): synchronous vs asynchronous follow-up TTFT."""
    out = {}
    for name, a, b in pairs:
        fu = [(d, t) for d, t in keys if t > 1 and (a, d, t) in rows and (b, d, t) in rows]
        if not fu:
            continue
        sa = np.array([rows[(a, d, t)]["ttft_ms"] for d, t in fu]); sb = np.array([rows[(b, d, t)]["ttft_ms"] for d, t in fu])
        out[name] = {"sync_mean_ms": float(sa.mean()), "async_mean_ms": float(sb.mean()), "paired_mean_difference_ms": float((sb - sa).mean()),
                     "requests": int(sa.size)}
    return out


def order_reference(rows, keys):
    """Reported, never a gate: requests whose tokens differ between Ours and the reordered packed arm."""
    out = {}
    for turn in (1, 2, 3):
        n = diff = same_prompt_diff = only_a = only_b = 0
        for dialog, t in keys:
            a, b = rows.get((H.OURS_ASYNC, dialog, t)), rows.get((H.REORDER_ASYNC, dialog, t))
            if t != turn or a is None or b is None:
                continue
            n += 1
            if tokens(a) != tokens(b):
                diff += 1; same_prompt_diff += a["prompt"] == b["prompt"]
            only_a += a["score"] > b["score"]; only_b += b["score"] > a["score"]
        out[f"t{turn}"] = {"requests": n, "tokens_differ": diff, "tokens_differ_with_equal_prompt": same_prompt_diff,
                           "only_ours_correct": only_a, "only_reordered_correct": only_b}
    return {"criterion": "none (reported only): a different key order changes the BF16 summation order, so boundary cases may flip",
            "by_turn": out, "status": "REPORTED"}


def gates_3_4_5(rows, keys):
    g3 = dict(compare(rows, H.OURS, H.OURS_ASYNC, keys, ("ssd_read_bytes", "packed_read_bytes")),
              criterion="Ours: generated tokens and bytes read identical between the synchronous and the asynchronous arm, every request")
    g4 = dict(compare(rows, H.OURS_ASYNC, H.OURS_REPEAT, keys),
              criterion="Ours executed twice on the same requests: generated tokens identical, every request")
    g5a = dict(compare(rows, H.NEW_R100, H.FULL_REORDERED, keys),
               criterion="IMPRESS at 100 % retention vs the FullLoad loader reading the same reordered store: generated tokens identical, every request")
    g5b = dict(compare(rows, H.NEW, H.NEW_ASYNC, keys, BYTE_FIELDS),
               criterion="IMPRESS: generated tokens and bytes read (total and by kind) identical between the synchronous and the asynchronous arm, every request")
    g5 = {"retention_100_vs_fullload_loader_on_reordered_store": g5a, "sync_vs_async": g5b,
          "status": "PASS" if g5a["status"] == g5b["status"] == "PASS" else "FAIL"}
    return g3, g4, g5


# ------------------------------------------------------------------ gate 7
def stage_tests(run, config):
    suites = [G.run_suite(sys.executable), G.run_suite(PY_CROSS)]
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", NEW_TEST], capture_output=True, text=True, cwd=ROOT)
    ran = re.search(r"^Ran (\d+) tests", p.stdout + p.stderr, re.M)
    ok = all(r["status"] == "PASS" for r in suites) and p.returncode == 0 and bool(ran)
    return receipt(run, "tests", {
        "gate": "gate7_existing_tests", "status": "PASS" if ok else "FAIL", "suites": suites,
        "new_unit_tests": {"file": f"tests/{NEW_TEST}", "ran": int(ran.group(1)) if ran else None, "ok": p.returncode == 0},
        "criterion": "no failing test other than the failures that existed before this work (transformers 4.57.6: the 3 GLM-4.6V tests; "
                     "transformers 5.13.1: 2 vision-capture tests); the new test file passes"}, config)


# ------------------------------------------------------ gates 1, 3-6 (4 images)
def stage_gate(run, config):
    manifest, keys, execution = phase_state(run, "gate", config)
    if execution["status"] != "PASS":
        return receipt(run, "gate", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "gate")
    g1 = gate1_store_contents(run, "gate", manifest)
    g3, g4, g5 = gates_3_4_5(rows, keys)
    g6 = gate6_recomp(rows, keys)
    parts = {"gate1_store_contents": g1, "gate3_ours_sync_equals_async": g3, "gate4_ours_repeat": g4, "gate5_impress": g5, "gate6_recomp_continuity": g6}
    failed = [k for k, v in parts.items() if v["status"] != "PASS"]
    return receipt(run, "gate", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "images": manifest["phase_images"]["gate"], "execution": execution, **parts,
        "reported_turn1_identity": turn1_identity(rows, keys, H.PHASE_METHODS["gate"]),
        "reported_ours_vs_reordered": order_reference(rows, keys),
        "reference_sync_vs_async_followup_ttft": sync_async_timing(rows, keys, (("IMPRESS", H.NEW, H.NEW_ASYNC), ("Ours", H.OURS, H.OURS_ASYNC)))}, config)


# ------------------------------------------------------------------- gate 2
def stage_qindep(run, config, phase):
    """Question independence of the Ours store (GPU).

    For every image of the phase, three NEW pixel requests with three different
    questions each build an Ours store through the run's own persistence path.

    Criterion (what is stored and where): the per-layer original-index tables,
    the row counts, the byte table, the file set and every metadata field that
    is not a record of the capturing request are identical across the three
    stores, and the index tables equal those of the store the phase committed.

    Reported, not judged (contract section 7, gate 2): the K/V payload bytes.
    The visual-prefix K/V captured in turn 1 can differ with the capturing
    prompt, in every store type; per image this stage records which pairs of
    questions have equal payloads, and -- from the same three captures -- the
    payload digests of the reordered store.
    """
    import torch
    from mmimpress.packed_store import PACKED_NAME, layer_original_indices, packed_file_sha256
    assert phase in ("gate", "devgate")
    name = f"qindep_{phase}"
    full_config, manifest_full = H.verify_freeze(run)
    manifest, keys, execution = phase_state(run, phase, config)
    if execution["status"] != "PASS":
        return receipt(run, name, {"status": "FAIL", "execution": execution}, config)
    torch.set_num_threads(full_config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234); torch.set_grad_enabled(False)
    H.assert_gpu_exclusive()
    runner, servers, _ = H.load_servers()
    images = []
    try:
        ids = set(manifest["phase_dialogue_ids"][phase])
        attempts = attempts_of(run, phase)
        for index, image in enumerate(manifest["phase_images"][phase]):
            folder = attempts[image]
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
            if scratch.exists():      # left by an interrupted earlier execution of this stage
                H.remove_scratch(run, scratch)
            scratch.mkdir(parents=True)
            builds = []
            for i, q in enumerate(questions):
                prompt = "USER: <image>\n" + "\n".join([f"Current question Q1: {q}", f"{H.SHORT_ANSWER} ASSISTANT:"])
                result, diag = H.normal_request(runner, servers["base"], path, lambda: (prompt, "", []), "packed")
                source = {"dialog_id": ds[0]["dialog_id"], "method_id": H.OURS_ASYNC, "physical_execution_id": f"qindep-{image}-{i}",
                          "logical_request_id": f"qindep:{run.name}:{phase}:{image}:q{i}"}
                # the same capture written in both row orders: the Ours store (judged) and the reordered store (reported)
                H.persist(runner, servers["base"], result, diag, scratch / f"q{i}", image, H.ORIG_KIND, source, full_config["layer_budgets"], None)
                H.persist(runner, servers["base"], result, diag, scratch / f"q{i}_reorder", image, H.PACKED_KIND, source, full_config["layer_budgets"], None)
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
            # the ids up to the end of the image (the stored prefix) never contain the question
            prefix_equal = all(b["meta"]["prefix_input_ids"] == ref["meta"]["prefix_input_ids"] for b in builds)
            files_equal = all(b["files"] == ref["files"] for b in builds) and "visionzip_layout.pt" not in ref["files"]
            flags = all(b["meta"].get("layout_uses_dataset_question") is False and b["meta"].get("llm_used_for_layout_scoring") is False
                        and b["meta"].get("calibration_questions") == 0 for b in builds)
            committed_tables_equal = (committed["stored_original_indices"] == ref["meta"]["stored_original_indices"]
                                      and committed["stored_min_keep_count"] == ref["meta"]["stored_min_keep_count"]
                                      and committed["layer_keep_counts"] == ref["meta"]["layer_keep_counts"]
                                      and committed["packed_layout"] == ref["meta"]["packed_layout"])
            ok = tables_equal and counts_equal and byte_table_equal and meta_equal and prefix_equal and files_equal and flags and committed_tables_equal
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
                           "index_tables_equal": tables_equal, "row_counts_equal": counts_equal, "byte_table_equal": byte_table_equal,
                           "metadata_equal_outside_request_records_and_payload_digest": meta_equal, "metadata_keys_that_differ": differing,
                           "stored_prefix_ids_equal": prefix_equal,
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
            H.remove_scratch(run, scratch)
            H.STATUS.update(run, stage=f"gate2_question_independence_{phase}", gate2_images_done=index + 1,
                            gate2_images=len(manifest["phase_images"][phase]))
            print(json.dumps({"image_id": image, "status": images[-1]["status"], "suffix_tokens": images[-1]["suffix_tokens"],
                              "payload_pairs": [(x["suffix_tokens"], x["payload_equal"], round(x["fraction_differ"], 4)) for x in pairs]}), flush=True)
    finally:
        for server in servers.values():
            server.close()
    all_pairs = [x for im in images for x in im["reported_payload"]["pairs"]]
    same_len = [x for x in all_pairs if x["equal_prompt_length"]]; other = [x for x in all_pairs if not x["equal_prompt_length"]]
    return receipt(run, name, {
        "gate": f"gate2_question_independent_{phase}",
        "status": "PASS" if images and len(images) == len(manifest["phase_images"][phase]) and all(x["status"] == "PASS" for x in images) else "FAIL",
        "criterion": "for every image, the Ours stores built from three different questions hold the same selection and layout: per-layer "
                     "original-index lists, row counts, byte table, file set, the stored prefix ids, and every metadata field except the record of "
                     "the capturing request and the payload digest; and the index tables equal those of the phase's committed store",
        "reported_not_judged": "the K/V payload bytes (see payload_summary and each image's reported_payload)",
        "inputs_of_the_store": "captured Turn-1 visual-prefix K/V, CLIP penultimate-layer CLS-to-patch attention, the frozen PYR-b20 layer budget; "
                               "no question text, no decoder attention, no calibration score",
        "payload_summary": {
            "question_pairs": len(all_pairs),
            "equal_prompt_length_pairs": len(same_len), "equal_prompt_length_pairs_with_equal_payload": sum(x["payload_equal"] for x in same_len),
            "different_prompt_length_pairs": len(other), "different_prompt_length_pairs_with_equal_payload": sum(x["payload_equal"] for x in other),
            "mean_fraction_of_fp16_elements_that_differ_when_lengths_differ": float(np.mean([x["fraction_differ"] for x in other])) if other else None,
            "max_abs_diff_when_lengths_differ": max((x["max_abs_diff"] for x in other), default=None),
            "pairs_where_reordered_store_agrees_with_ours_store_on_equal_or_different": sum(x["payload_equal"] == x["reorder_payload_equal"] for x in all_pairs),
            "images_with_all_three_payloads_equal": sum(im["reported_payload"]["all_three_payloads_equal"] for im in images),
            "images_whose_first_question_payload_equals_committed_store": sum(im["reported_payload"]["first_question_payload_equals_committed_store"] for im in images),
            "images_with_equal_system_kv": sum(im["reported_payload"]["system_kv_equal"] for im in images)},
        "images": images, "image_count": len(images), "pass_count": sum(x["status"] == "PASS" for x in images)}, config)


# ------------------------------------------------------ gates 3, 4, 5 (DEV 40)
def stage_devgate(run, config):
    manifest, keys, execution = phase_state(run, "devgate", config)
    if execution["status"] != "PASS":
        return receipt(run, "devgate", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "devgate")
    g3, g4, g5 = gates_3_4_5(rows, keys)
    parts = {"gate3_ours_sync_equals_async": g3, "gate4_ours_repeat": g4, "gate5_impress": g5}
    failed = [k for k, v in parts.items() if v["status"] != "PASS"]
    return receipt(run, "devgate", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "image_count": len(manifest["phase_images"]["devgate"]), "execution": execution, **parts,
        "reported_turn1_identity": turn1_identity(rows, keys, H.PHASE_METHODS["devgate"]),
        "reference_sync_vs_async_followup_ttft": sync_async_timing(rows, keys, (("IMPRESS", H.NEW, H.NEW_ASYNC), ("Ours", H.OURS, H.OURS_ASYNC)))}, config)


# ------------------------------------------------------ gates 1, 6, 8 (DEV 40)
def stage_dev(run, config):
    manifest, keys, execution = phase_state(run, "dev", config)
    if execution["status"] != "PASS":
        return receipt(run, "dev", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "dev")
    g1 = gate1_store_contents(run, "dev", manifest)
    g6 = gate6_recomp(rows, keys)
    parts = {"gate1_store_contents": g1, "gate6_recomp_continuity": g6}
    failed = [k for k, v in parts.items() if v["status"] != "PASS"]
    seconds = []
    for image, attempt in attempts_of(run, "dev").items():
        seconds.append((attempt / "cleanup_receipt.json").stat().st_mtime - (attempt / "capacity_before.json").stat().st_mtime)
    return receipt(run, "dev", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "image_count": len(manifest["phase_images"]["dev"]),
        "execution": dict(execution, gate="gate8_dev_pilot_executed"), **parts,
        "reported_turn1_identity": turn1_identity(rows, keys, H.PHASE_METHODS["dev"]),
        "reported_ours_vs_reordered": order_reference(rows, keys),
        "pilot_seconds_per_image_mean": float(np.mean(seconds)), "pilot_seconds_total": float(np.sum(seconds))}, config)


# -------------------------------------------------------------------- admit
def stage_admit(run, config):
    steps = {}
    for name in REQUIRED:
        path = run / "gates" / f"{name}.json"
        doc = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        ok = doc.get("status") == "PASS" and doc.get("config_sha256") == config["_file_sha256"]
        steps[name] = {"status": doc.get("status"), "bound_to_this_freeze": doc.get("config_sha256") == config["_file_sha256"], "ok": ok,
                       "failed": doc.get("failed")}
    status = "PASS" if all(s["ok"] for s in steps.values()) else "FAIL"
    return receipt(run, "admission", {
        "status": status, "steps": steps,
        "rule": "the main phase starts only if gates 1-7 are PASS on the 4 gate images and on the 40 DEV images and the DEV pilot (gate 8) "
                "was executed; otherwise the driver records the failure and stops"}, config)


def stage_posthoc(run, config):
    """After the main phase; reported, not a gate."""
    manifest, keys, execution = phase_state(run, "main", config)
    rows = committed_rows(run, "main")
    g6 = gate6_recomp(rows, keys)
    t1 = turn1_identity(rows, keys, H.PHASE_METHODS["main"])
    out = {"status": "REPORTED", "execution": execution, "recomp_continuity_main": g6, "turn1_identity_main": t1,
           "config_sha256": config["_file_sha256"], "written_unix": time.time()}
    atomic_json(run / "gates" / "posthoc_main.json", out, replace=True)
    print(json.dumps({"requests": g6["requests"], "recomp_mismatch": g6["mismatch_count"], "turn1_mismatch_dialogues": t1["mismatch_count"]}))
    return dict(out, status="PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "gate", "qindep", "devgate", "dev", "admit", "posthoc"), required=True)
    p.add_argument("--phase", choices=("gate", "devgate"), default="gate", help="qindep: which phase's images")
    args = p.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith(H.RUN_PREFIX)
    config = frozen(run)
    changed = H.freeze_violations(config["source_sha256"])
    if changed and "_dev_" not in run.name:
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
    doc = {"tests": lambda: stage_tests(run, config), "gate": lambda: stage_gate(run, config),
           "qindep": lambda: stage_qindep(run, config, args.phase), "devgate": lambda: stage_devgate(run, config),
           "dev": lambda: stage_dev(run, config), "admit": lambda: stage_admit(run, config), "posthoc": lambda: stage_posthoc(run, config)}[args.stage]()
    sys.exit(0 if doc["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

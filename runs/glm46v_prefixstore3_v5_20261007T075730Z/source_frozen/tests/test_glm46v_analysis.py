"""CPU-only independent analysis contracts; synthetic numbers are not GPU results."""
from copy import deepcopy
import unittest

from mmimpress.glm46v.analysis import (
    METHODS, NATIVE_HISTORY_POLICY, analyze_rows, audit_requests,
    independent_native_body, paired_cluster_ci, pinned_history_decoder,
)


def fixture():
    ds = []
    for did, iid in (("d1", "a"), ("d2", "b"), ("d3", "b")):
        ds.append(dict(dialog_id=did, image_id=iid,
                       turns=[dict(turn_id=t, question_id=f"{did}q{t}",
                                   question=f"question {did} {t}", answers=["yes"])
                              for t in (1, 2, 3)]))
    manifest = dict(dialogues=ds)
    rows = []
    for ordinal, d in enumerate(ds):
        histories = {m: [] for m in METHODS}
        lineage = {m: [] for m in METHODS}
        for q in d["turns"]:
            for method in METHODS:
                t, iid, did = q["turn_id"], d["image_id"], d["dialog_id"]
                pixel = t == 1 or method == "recompute"
                reads = 0 if pixel else 32 if method == "fullload" else 8
                ttft = 20. if method == "recompute" else 15. if method == "fullload" else 10. if iid == "a" else 18.
                prediction = "no" if method == METHODS[-1] and iid == "b" else "yes"
                physical, logical = f"physical-{did}-{method}-{t}", f"logical-{did}-{method}-{t}"
                row = dict(image_id=iid, dialog_id=did, turn_id=t,
                           method_id=method, question_id=q["question_id"],
                           question=q["question"], gold=q["answers"],
                           prediction=prediction, phase="main", status="PASS",
                           physical_execution_id=physical, logical_request_id=logical,
                           method_order=list(METHODS[ordinal % 3:]) + list(METHODS[:ordinal % 3]),
                           input_ids=[[1, 2]], prompt="synthetic CPU fixture",
                           generated_token_ids=[3, 4],
                           history=deepcopy(histories[method]), history_entries=deepcopy(lineage[method]),
                           ttft_ms=ttft, request_e2e_ms=ttft + 1,
                           ssd_read_bytes=reads, ssd_preads=int(not pixel),
                           os_pread_trace=[] if pixel else [dict(source=f"layer_000/{kind}.bin", kind="content", offset=0, requested=reads // 2, returned=reads // 2) for kind in ("k", "v")],
                           N_content=8, k=8 if pixel or method == "fullload" else 2,
                           selected_content_original=[0, 1], S_structural=0,
                           kv_geometry=dict(num_layers=1, num_kv_heads=1, head_dim=1, dtype_bytes=2))
                row["ssd_preads"] = 0 if pixel else 2
                if not pixel:
                    source_did = "d1" if iid == "a" else "d2"
                    row["source_store"] = dict(path=f"synthetic-{iid}-{method}", identity={"synthetic": True},
                                              source=dict(dialog_id=source_did, method_id=method,
                                                          physical_execution_id=f"physical-{source_did}-{method}-1",
                                                          logical_request_id=f"logical-{source_did}-{method}-1"))
                rows.append(row)
                histories[method].append([q["question"], prediction])
                lineage[method].append(dict(source_method_id=method, source_dialog_id=did,
                                            source_physical_execution_id=physical,
                                            source_logical_request_id=logical))
    events = []
    for iid, did in (("a", "d1"), ("b", "d2")):
        for method in METHODS[1:]:
            for kind, duration in (("build", 5.), ("activation", 2.)):
                events.append(dict(image_id=iid, method_id=method, source_dialog_id=did,
                                   event_kind=kind, event_id=f"{iid}-{method}-{kind}",
                                   duration_ms=duration, endpoint="outer_writer_return",
                                   file_logical_bytes=400 if method == "fullload" else 100,
                                   allocated_bytes=4096, os_write_bytes=400 if method == "fullload" else 100,
                                   N_content=8, stored_k=8 if method == "fullload" else 2))
    return manifest, rows, events


class FixtureTokenizer:
    # A decoder oracle independent of production GLM history/answer helpers.
    tokens = {10: "yes", 11: "no", 151361: "<|begin_of_box|>",
              151362: "<|end_of_box|>", 151336: "<|user|>",
              151348: "</tool_response>", 12: "</think>", 13: " "}

    def decode(self, ids, *, skip_special_tokens=False):
        if skip_special_tokens:
            raise AssertionError("native body must retain internal control tokens")
        return "".join(self.tokens[i] for i in ids)


def native_fixture():
    manifest, rows, events = fixture()
    for row in rows:
        word_id = 10 if row["prediction"] == "yes" else 11
        body = "<|begin_of_box|>" + row["prediction"] + "<|end_of_box|>"
        row.update(generated_token_ids=[151361, word_id, 151362, 151336],
                   generated_text_with_special_tokens=body + "<|user|>",
                   history_text=body, history_token_ids=[151361, word_id, 151362],
                   history_native_policy=NATIVE_HISTORY_POLICY)
    refresh_native_history(rows)
    return manifest, rows, events


def refresh_native_history(rows):
    by = {(r["dialog_id"], r["method_id"], r["turn_id"]): r for r in rows}
    for row in rows:
        prior = [by[row["dialog_id"], row["method_id"], t] for t in range(1, row["turn_id"])]
        row["history"] = [[p["question"], p["history_text"]] for p in prior]
        for entry, previous in zip(row["history_entries"], prior):
            entry.update(history_text=previous["history_text"],
                         history_token_ids=list(previous["history_token_ids"]),
                         generated_token_ids=list(previous["generated_token_ids"]))


class IndependentAnalysisTest(unittest.TestCase):
    def test_native_history_preserves_box_body_but_scores_final_answer(self):
        manifest, rows, events = native_fixture()
        result = analyze_rows(rows, manifest, events, tokenizer=FixtureTokenizer(),
                              eos_token_ids={151336, 151348})
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["audit"]["history_native_policy"], NATIVE_HISTORY_POLICY)
        quality = next(q for q in result["quality"] if q["method_id"] == "recompute" and q["scope"] == "all")
        self.assertEqual(quality["accuracy"], 100.)
        self.assertNotEqual(rows[3]["history"][0][1], rows[0]["prediction"])

    def test_parsed_answer_history_and_missing_body_are_rejected(self):
        manifest, rows, _ = native_fixture()
        changed = deepcopy(rows)
        changed[3]["history"][0][1] = changed[0]["prediction"]
        bad = audit_requests(changed, manifest, tokenizer=FixtureTokenizer(), eos_token_ids={151336})
        self.assertIn("method_local_generated_history", {f["check"] for f in bad["failures"]})
        changed = deepcopy(rows)
        del changed[0]["history_text"]
        bad = audit_requests(changed, manifest, tokenizer=FixtureTokenizer(), eos_token_ids={151336})
        self.assertEqual(bad["status"], "FAIL")
        self.assertGreater(bad["missing_evidence"]["history_text"], 0)

    def test_native_body_tampering_or_missing_boundary_ids_is_rejected(self):
        manifest, rows, _ = native_fixture()
        for field, value, check in (
                ("history_text", "yes", "native_body_independent_decode"),
                ("history_token_ids", [10], "native_body_actual_generated_ids"),
                ("generated_text_with_special_tokens", "yes", "native_full_generation_independent_decode"),
                ("history_native_policy", "unknown-policy", "native_history_policy")):
            with self.subTest(field=field):
                changed = deepcopy(rows)
                changed[0][field] = value
                bad = audit_requests(changed, manifest, tokenizer=FixtureTokenizer(), eos_token_ids={151336})
                self.assertIn(check, {f["check"] for f in bad["failures"]})

    def test_native_lineage_body_must_match_its_own_generated_ids(self):
        manifest, rows, _ = native_fixture()
        changed = deepcopy(rows)
        changed[3]["history_entries"][0]["history_token_ids"] = [151361, 11, 151362]
        bad = audit_requests(changed, manifest, tokenizer=FixtureTokenizer(), eos_token_ids={151336})
        self.assertIn("native_body_history_lineage_ids", {f["check"] for f in bad["failures"]})

    def test_native_malformed_output_is_measured_and_not_censored(self):
        manifest, rows, _ = native_fixture()
        # Native decoder emitted a closing think marker without an opening one.
        body = "yes</think>yes<|end_of_box|>"
        rows[0].update(prediction=body, history_text=body,
                       generated_token_ids=[10, 12, 10, 151362, 151336],
                       history_token_ids=[10, 12, 10, 151362],
                       generated_text_with_special_tokens=body + "<|user|>",
                       reasoning_leak=True, native_output_anomaly_observed=True)
        refresh_native_history(rows)
        audit = audit_requests(rows, manifest, tokenizer=FixtureTokenizer(), eos_token_ids={151336})
        self.assertEqual(audit["status"], "PASS")

    def test_only_terminal_eos_removed_and_exact_whitespace_retained(self):
        body, ids = independent_native_body([13, 151336, 10, 13, 151348],
                                            FixtureTokenizer(), {151336, 151348})
        self.assertEqual(ids, [13, 151336, 10, 13])
        self.assertEqual(body, " <|user|>yes ")
        plain, plain_ids = independent_native_body([13, 10, 13], FixtureTokenizer(), {151336})
        self.assertEqual((plain, plain_ids), (" yes ", [13, 10, 13]))
        empty, empty_ids = independent_native_body([151336], FixtureTokenizer(), {151336})
        self.assertEqual((empty, empty_ids), ("", []))

    def test_real_pinned_tokenizer_and_all_four_native_eos_ids(self):
        tokenizer, eos = pinned_history_decoder()
        self.assertEqual(eos, frozenset({151329, 151336, 151338, 151348}))
        body, ids = independent_native_body([151361, 34, 4830, 151362, 151336], tokenizer, eos)
        self.assertEqual(body, "<|begin_of_box|>Crown<|end_of_box|>")
        self.assertEqual(ids, [151361, 34, 4830, 151362])

    def test_unequal_image_clusters_preserve_request_weight(self):
        values = [dict(image_id="a", delta=-10.)] + [dict(image_id="b", delta=-2.)] * 2
        ci = paired_cluster_ci(values, "delta", ["a", "b"])
        self.assertAlmostEqual(ci["point"], -14 / 3)
        self.assertEqual((ci["ci_low"], ci["ci_high"]), (-10., -2.))
        self.assertNotEqual(ci["point"], -6.)

    def test_real_stream_attribution_and_derived_session(self):
        manifest, rows, events = fixture()
        report = analyze_rows(rows, manifest, events)
        self.assertEqual(report["status"], "PASS")
        audit = report["session_accounting"]
        self.assertEqual(audit["physical_setup_ms"], 28.)
        self.assertEqual(audit["attributed_setup_ms"], 28.)
        followup = next(r for r in report["paired_analysis"] if r["baseline"] == "recompute"
                        and r["scope"] == "followup" and r["metric"] == "ttft_ms")
        self.assertAlmostEqual(followup["point"], -14 / 3)
        b3 = next(s for s in report["session"] if s["dialog_id"] == "d3" and s["method_id"] == METHODS[-1])
        self.assertEqual(b3["actual_build_attribution_ms"], 0.)
        self.assertEqual(b3["derived_standalone_session_ms"] - b3["physical_stream_session_ms"], 7.)

    def test_negative_membership_history_and_io(self):
        manifest, rows, _ = fixture()
        for mutation, expected in (
            (lambda rs: rs[0].update(gold=["changed"]), "unchanged_gold"),
            (lambda rs: rs[9].update(history=[["question d1 1", "gold injected"]]), "method_local_generated_history"),
            (lambda rs: rs[4]["os_pread_trace"][0].update(returned=15), "OS_returned_byte_sum"),
            (lambda rs: rs.append(deepcopy(rs[0])), "duplicate_final_request"),
        ):
            with self.subTest(expected=expected):
                changed = deepcopy(rows)
                mutation(changed)
                audit = audit_requests(changed, manifest)
                self.assertEqual(audit["status"], "FAIL")
                self.assertIn(expected, {f["check"] for f in audit["failures"]})

    def test_same_image_dialogues_cannot_reselect(self):
        manifest, rows, _ = fixture()
        changed = deepcopy(rows)
        target = next(r for r in changed if r["dialog_id"] == "d3" and r["method_id"] == METHODS[-1] and r["turn_id"] == 2)
        target["selected_content_original"] = [1, 2]
        audit = audit_requests(changed, manifest)
        self.assertIn("same_image_frozen_selection", {f["check"] for f in audit["failures"]})

    def test_read_equal_bytes_wrong_offset_rejected(self):
        manifest, rows, _ = fixture()
        changed = deepcopy(rows)
        changed[4]["os_pread_trace"][0]["offset"] = 2
        audit = audit_requests(changed, manifest)
        self.assertIn("sequential_once_payload_read", {f["check"] for f in audit["failures"]})

    def test_foreign_source_and_changed_store_identity_rejected(self):
        manifest, rows, _ = fixture()
        changed = deepcopy(rows)
        target = next(r for r in changed if r["dialog_id"] == "d3" and r["method_id"] == METHODS[-1] and r["turn_id"] == 2)
        target["source_store"]["source"]["dialog_id"] = "d3"
        target["source_store"]["identity"] = {"foreign_image": True}
        audit = audit_requests(changed, manifest)
        checks = {f["check"] for f in audit["failures"]}
        self.assertIn("image_source_T1_lineage", checks)
        self.assertIn("same_image_store_identity", checks)

    def test_inner_persistence_receipt_rejected(self):
        manifest, rows, events = fixture()
        events[0]["endpoint"] = "inner_durable_publication"
        with self.assertRaisesRegex(ValueError, "outer writer-return"):
            analyze_rows(rows, manifest, events)

    def test_empty_main_has_no_synthetic_accuracy_or_latency(self):
        manifest, _, _ = fixture()
        report = analyze_rows([], manifest)
        self.assertEqual(report["status"], "NOT_RUN")
        self.assertEqual(report["paired_analysis"], [])
        self.assertTrue(all(q["accuracy"] == "NOT_MEASURED" for q in report["quality"]))
        self.assertTrue(all(l["ttft_ms_mean"] == "NOT_MEASURED" for l in report["latency"]))


if __name__ == "__main__":
    unittest.main()

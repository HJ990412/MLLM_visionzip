"""CPU-only independent analysis contracts; synthetic numbers are not GPU results."""
from copy import deepcopy
import unittest

from mmimpress.glm46v.analysis import (
    METHODS, analyze_rows, audit_requests, paired_cluster_ci,
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


class IndependentAnalysisTest(unittest.TestCase):
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

"""Independent GLM raw analysis; no model, production scorer, or store import.

All result values come from adopted raw requests and explicit setup receipts.
This module never labels absent observations as zero or runs quality-driven
selection. Native store correctness remains a separate required GPU gate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

METHODS = ("recompute", "fullload", "ours_prefixstore_kv25")
NOT_MEASURED = "NOT_MEASURED"
EXPECTED_MAIN_HASH = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_answer(value: Any) -> str:
    """The existing strict normalized EM rule, independently implemented."""
    words = re.sub(r"[^\w\s]", " ", str(value).lower()).split()
    return " ".join(w for w in words if w not in {"a", "an", "the"})


def jsonl(path: Path | str) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def workload_hash(dialogues: Sequence[Mapping]) -> str:
    raw = "".join(f"{d.get('dialog_id')}\t{t['turn_id']}\t{t['question_id']}\n"
                  for d in dialogues for t in d["turns"]).encode()
    return hashlib.sha256(raw).hexdigest()


def statistics(values: Iterable[Any]) -> dict:
    # bool is not a timing or byte measurement; NaN/inf must never enter tables.
    x = [float(v) for v in values if isinstance(v, (int, float))
         and not isinstance(v, bool) and math.isfinite(v)]
    if not x:
        return dict(mean=NOT_MEASURED, p50=NOT_MEASURED, p95=NOT_MEASURED,
                    measured_count=0)
    a = np.asarray(x, dtype=np.float64)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), measured_count=len(x))


def paired_cluster_ci(rows: Sequence[Mapping], field: str, images: Sequence[str],
                      *, seed: int = 1234, resamples: int = 10000) -> dict:
    """Resample whole image clusters and pool observation sums/counts.

    Rows must already be matched method differences. Pairing before sampling
    preserves every method, turn and dialogue within the same image draw.
    """
    if not rows:
        return dict(point=NOT_MEASURED, ci_low=NOT_MEASURED,
                    ci_high=NOT_MEASURED, observations=0, images=0,
                    seed=seed, resamples=resamples)
    images = list(images)
    if len(set(images)) != len(images) or not images or resamples < 1:
        raise ValueError("invalid image bootstrap population")
    indices = {im: i for i, im in enumerate(images)}
    sums = np.zeros(len(images), dtype=np.float64)
    counts = np.zeros(len(images), dtype=np.int64)
    for row in rows:
        v = row[field]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("bootstrap requires finite measured differences")
        i = indices[row["image_id"]]
        sums[i] += float(v)
        counts[i] += 1
    if (counts == 0).any():
        raise ValueError("bootstrap images lack matched observations")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    # Bound temporary index storage even for larger future image populations.
    for start in range(0, resamples, 256):
        end = min(start + 256, resamples)
        ix = rng.integers(0, len(images), size=(end - start, len(images)))
        estimates[start:end] = sums[ix].sum(axis=1) / counts[ix].sum(axis=1)
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return dict(point=float(sums.sum() / counts.sum()), ci_low=float(lo),
                ci_high=float(hi), observations=int(counts.sum()),
                images=len(images), seed=seed, resamples=resamples,
                weighting="pooled observation sums/counts; paired image cluster",
                interval="percentile 95%; includes zero does not imply equivalence")


def _dialog_id(row: Mapping) -> str:
    return str(row.get("dialog_id") or f"gqa:{row['image_id']}")


def _field(row: Mapping, name: str, default=None):
    if name in row:
        return row[name]
    result = row.get("result", {})
    if name in result:
        return result[name]
    io = row.get("read_io", result.get("read_io", {})) or {}
    if name == "ssd_preads":
        return io.get("preads", default)
    if name == "ssd_read_bytes":
        return io.get("bytes", default)
    timings = row.get("timing_ms", result.get("timing_ms", {})) or {}
    timing_aliases = {"cache_assembly_ms": "assembly_ms"}
    if name in timings:
        return timings[name]
    if name in timing_aliases:
        return timings.get(timing_aliases[name], default)
    return default


def _trace(row: Mapping):
    trace = _field(row, "os_pread_trace")
    if trace is None:
        io = _field(row, "read_io", {}) or {}
        trace = io.get("call_details", io.get("trace", io.get("events", io.get("ranges"))))
    return trace


def _trace_returned(event: Mapping) -> int:
    for k in ("returned", "actual_bytes", "returned_bytes", "bytes_returned"):
        if k in event:
            return int(event[k])
    raise ValueError("pread trace lacks actual returned bytes")


def _expected(manifest: Mapping, methods: Sequence[str]) -> dict:
    entries = manifest.get("dialogues", manifest.get("images"))
    if not isinstance(entries, list):
        raise ValueError("manifest requires ordered dialogues/images")
    expected = {}
    for ordinal, entry in enumerate(entries):
        canonical_ordinal = entry.get("canonical_ordinal", ordinal)
        if not isinstance(canonical_ordinal, int) or isinstance(canonical_ordinal, bool) or canonical_ordinal < 0:
            raise ValueError("manifest canonical ordinal must be a nonnegative integer")
        did = _dialog_id(entry)
        for turn in entry["turns"]:
            for method in methods:
                key = (did, method, int(turn["turn_id"]))
                if key in expected:
                    raise ValueError("duplicate manifest request")
                expected[key] = dict(image_id=str(entry["image_id"]),
                                     entry=entry, turn=turn, ordinal=canonical_ordinal)
    return expected


def audit_requests(rows: Sequence[Mapping], manifest: Mapping, *, phase="main",
                   methods: Sequence[str] = METHODS) -> dict:
    expected = _expected(manifest, methods)
    failures = []
    missing_evidence = Counter()
    seen = {}
    physical, logical = set(), set()
    selected_by_image = defaultdict(list)
    store_identity_by_image = {}
    history_none = phase == "gqa_pilot" or manifest.get("history") == "none"
    source_dialogues = {}
    for ex in expected.values():
        source_dialogues.setdefault(ex["image_id"], _dialog_id(ex["entry"]))

    def require(condition, label, key):
        if not condition:
            failures.append(dict(check=label, request=list(key)))

    for r in rows:
        try:
            key = (_dialog_id(r), str(r["method_id"]), int(r["turn_id"]))
            require(key not in seen, "duplicate_final_request", key)
            seen[key] = r
            ex = expected.get(key)
            require(ex is not None, "foreign_membership", key)
            if ex is None:
                continue
            t, m = key[2], key[1]
            q = ex["turn"]
            gold = q.get("answers", [q["gold"]] if "gold" in q else None)
            require(str(r["image_id"]) == ex["image_id"], "image_membership", key)
            require(str(r.get("question_id")) == str(q["question_id"]), "question_id", key)
            require(r.get("question") == q["question"], "question_text", key)
            require(r.get("gold") == gold, "unchanged_gold", key)
            require(r.get("phase", phase) == phase, "phase", key)
            require(r.get("status", "PASS") == "PASS", "adopted_request_status", key)
            require(isinstance(r.get("prediction"), str), "prediction_text", key)
            if "score" in r and gold:
                score = float(normalized_answer(r["prediction"]) == normalized_answer(gold[0]))
                require(r["score"] == score, "independent_strict_em", key)
            for name, pool in (("physical_execution_id", physical), ("logical_request_id", logical)):
                if name not in r:
                    missing_evidence[name] += 1
                else:
                    require(r[name] not in pool, "duplicate_" + name, key)
                    pool.add(r[name])
            if "method_order" in r:
                n = ex["ordinal"] % len(methods)
                order = list(methods[n:]) + list(methods[:n])
                require(r["method_order"] == order, "frozen_method_rotation", key)
            else:
                missing_evidence["method_order"] += 1
            if "canonical_ordinal" in r:
                require(r["canonical_ordinal"] == ex["ordinal"], "canonical_dialogue_ordinal", key)
            for name in ("input_ids", "prompt", "generated_token_ids"):
                if _field(r, name) is None:
                    missing_evidence[name] += 1
            ttft, e2e = _field(r, "ttft_ms"), _field(r, "request_e2e_ms")
            require(isinstance(ttft, (int, float)) and isinstance(e2e, (int, float))
                    and math.isfinite(ttft) and math.isfinite(e2e) and 0 <= ttft <= e2e,
                    "finite_timing_order", key)
            bytes_read = _field(r, "ssd_read_bytes")
            require(isinstance(bytes_read, int) and not isinstance(bytes_read, bool)
                    and bytes_read >= 0, "actual_read_bytes", key)
            trace = _trace(r)
            if trace is None:
                missing_evidence["os_pread_trace"] += 1
            else:
                require(sum(_trace_returned(e) for e in trace) == bytes_read,
                        "OS_returned_byte_sum", key)
                for e in trace:
                    require(int(e.get("offset", 0)) >= 0 and _trace_returned(e) >= 0,
                            "valid_pread_range", key)
                    requested = e.get("requested", e.get("requested_bytes"))
                    if requested is not None:
                        require(0 < _trace_returned(e) <= requested,
                                "invalid_or_EOF_pread", key)
                calls = _field(r, "ssd_preads", _field(r, "pread_calls"))
                if calls is not None:
                    require(calls == len(trace), "actual_pread_count", key)
            pixel = t == 1 or m == "recompute"
            if pixel:
                require(bytes_read == 0 and not trace, "pixels_no_KV_store_reads", key)
            n, k = _field(r, "N_content"), _field(r, "k")
            if not isinstance(n, int) or not isinstance(k, int):
                missing_evidence["original_N_budget"] += 1
            elif not pixel:
                require(n > 0 and k == ((n + 3) // 4 if m == methods[-1] else n),
                        "original_N_budget", key)
                geometry = _field(r, "kv_geometry")
                structural = _field(r, "S_structural")
                if not isinstance(geometry, dict) or not isinstance(structural, int):
                    missing_evidence["KV_geometry_for_independent_IO"] += 1
                elif trace is not None:
                    layers = int(geometry["num_layers"])
                    row_bytes = int(geometry["num_kv_heads"]) * int(geometry["head_dim"]) * int(geometry["dtype_bytes"])
                    require(layers > 0 and row_bytes > 0 and structural >= 0,
                            "native_KV_geometry", key)
                    expected_files = {f"layer_{li:03d}/{kind}.bin": k * row_bytes
                                      for li in range(layers) for kind in ("k", "v")}
                    if structural:
                        expected_files["structural_kv.bin"] = 2 * layers * structural * row_bytes
                    actual_files = defaultdict(list)
                    for e in trace:
                        path = Path(e.get("source", e.get("path", "UNKNOWN")))
                        rel = "/".join(path.parts[-2:]) if path.parent.name.startswith("layer_") else path.name
                        actual_files[rel].append((int(e["offset"]), _trace_returned(e)))
                    require(set(actual_files) == set(expected_files), "only_required_payload_reads", key)
                    for rel, length in expected_files.items():
                        cursor = 0
                        for offset, returned in actual_files.get(rel, []):
                            require(offset == cursor, "sequential_once_payload_read", key)
                            cursor += returned
                        require(cursor == length, "exact_native_payload_read_range", key)
            if m == methods[-1] and not pixel:
                selected = _field(r, "selected_content_original",
                                  _field(r, "selected_visual_original"))
                if selected is None:
                    missing_evidence["selected_content_original"] += 1
                else:
                    require(isinstance(n, int) and isinstance(k, int)
                            and len(selected) == k and len(set(selected)) == k
                            and all(isinstance(i, int) and 0 <= i < n for i in selected),
                            "selected_original_ids", key)
                    selected_by_image[str(r["image_id"])].append(tuple(sorted(selected)))
        except (KeyError, TypeError, ValueError) as e:
            failures.append(dict(check="malformed_raw", error=str(e)))
    for key, r in seen.items():
        did, m, t = key
        if key not in expected:
            continue
        history = []
        if not history_none:
            for pt in range(1, t):
                prev = seen.get((did, m, pt))
                if prev is None:
                    require(False, "missing_own_history_source", key)
                    continue
                history.append([expected[did, m, pt]["turn"]["question"], prev["prediction"]])
        if "history" in r:
            require([list(x) for x in r["history"]] == history,
                    "method_local_generated_history", key)
        else:
            missing_evidence["history"] += 1
        if "history_entries" in r:
            entries = r["history_entries"]
            require(len(entries) == len(history), "history_lineage_length", key)
            for pt, entry in enumerate(entries, 1):
                prev = seen.get((did, m, pt), {})
                require(entry.get("source_method_id", entry.get("method_id")) == m
                        and str(entry.get("source_dialog_id", entry.get("dialog_id"))) == did
                        and entry.get("source_physical_execution_id") == prev.get("physical_execution_id")
                        and entry.get("source_logical_request_id") == prev.get("logical_request_id"),
                        "history_lineage", key)
        elif not history_none and t > 1:
            missing_evidence["history_entries"] += 1
        if t > 1 and m != methods[0]:
            store = r.get("source_store")
            if not isinstance(store, dict) or not store.get("path") or not store.get("identity"):
                missing_evidence["source_store_identity"] += 1
            else:
                source = store.get("source", {})
                identity_key = (str(r["image_id"]), m)
                identity = canonical_hash(store["identity"])
                old_identity = store_identity_by_image.setdefault(identity_key, identity)
                require(old_identity == identity, "same_image_store_identity", key)
                source_did = source_dialogues[str(r["image_id"])]
                source_row = seen.get((source_did, m, 1), {})
                actual_did = str(source.get("dialog_id") or f"gqa:{r['image_id']}")
                require(actual_did == source_did and source.get("method_id") == m
                        and source.get("physical_execution_id") == source_row.get("physical_execution_id")
                        and source.get("logical_request_id") == source_row.get("logical_request_id"),
                        "image_source_T1_lineage", key)
    for iid, choices in selected_by_image.items():
        if any(ids != choices[0] for ids in choices):
            failures.append(dict(check="same_image_frozen_selection", image_id=iid))
    missing = set(expected) - set(seen)
    complete = not missing and len(rows) == len(expected)
    return dict(status="FAIL" if failures else "NOT_RUN" if not rows
                else "INCOMPLETE" if missing_evidence or not complete else "PASS",
                requests=len(rows), expected_requests=len(expected),
                coverage_complete=complete, missing_requests=len(missing),
                missing_evidence=dict(missing_evidence), failures=failures,
                workload_ordered_sha256=workload_hash(manifest.get("dialogues", manifest.get("images"))),
                images=len({str(r.get("image_id")) for r in rows}),
                actual_generated_history=True,
                source="independent raw membership/history/selection/I/O audit",
                note="Native KV/logits/position/attention correctness is a separate required gate.")


def _setup(events: Sequence[Mapping], methods: Sequence[str]):
    """Explicit physical setup, using outer writer-return durations."""
    grouped = defaultdict(list)
    event_ids = set()
    for event in events:
        kind = event.get("event_kind")
        if kind not in {"build", "physical_build", "activation"}:
            raise ValueError("unknown setup event kind")
        eid = event["event_id"]
        if eid in event_ids:
            raise ValueError("duplicate physical setup event")
        event_ids.add(eid)
        method = event["method_id"]
        if method not in methods[1:]:
            raise ValueError("foreign method / ReComp KV setup")
        duration = event.get("outer_writer_wall_ms", event.get("duration_ms"))
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("setup lacks measured outer duration")
        if kind != "activation" and "outer_writer_wall_ms" not in event and event.get("endpoint") != "outer_writer_return":
            raise ValueError("build must declare measured outer writer-return endpoint")
        grouped[str(event["image_id"]), method].append({**event, "duration_ms": float(duration)})
    setup, storage = {}, []
    for (iid, method), es in grouped.items():
        builds = [e for e in es if e["event_kind"] != "activation"]
        acts = [e for e in es if e["event_kind"] == "activation"]
        if len(builds) != 1 or len(acts) != 1:
            raise ValueError("adopted image/method needs exactly one build and activation")
        build, activation = builds[0], acts[0]
        source = build["source_dialog_id"]
        setup[iid, method] = dict(source_dialog_id=str(source or f"gqa:{iid}"),
                                 build_ms=build["duration_ms"],
                                 activation_ms=activation["duration_ms"],
                                 setup_ms=build["duration_ms"] + activation["duration_ms"])
        fields = ("N_content", "stored_k", "file_logical_bytes", "allocated_bytes",
                  "os_write_bytes", "os_write_calls", "content_payload_bytes",
                  "structural_payload_bytes", "padding_bytes", "metadata_bytes")
        values = {k: build.get(k, NOT_MEASURED) for k in fields}
        values["file_logical_bytes"] = build.get("file_logical_bytes", build.get("logical_bytes", NOT_MEASURED))
        storage.append(dict(image_id=iid, method_id=method, build_event_id=build["event_id"],
                            activation_event_id=activation["event_id"],
                            persistence_ms=build["duration_ms"],
                            activation_ms=activation["duration_ms"], provenance="MEASURED",
                            **values))
    return setup, storage


def analyze_rows(rows: Sequence[Mapping], manifest: Mapping, setup_events=(), *,
                 phase="main", methods: Sequence[str] = METHODS,
                 seed=1234, resamples=10000) -> dict:
    methods = tuple(methods)
    if len(methods) != 3 or len(set(methods)) != 3:
        raise ValueError("exactly three distinct methods required")
    audit = audit_requests(rows, manifest, phase=phase, methods=methods)
    if audit["failures"]:
        # Do not publish quality tables based on invalid/adversarial membership.
        return dict(status="FAIL", audit=audit, quality=[], latency=[], storage=[],
                    session=[], paired_analysis=[], prediction_agreement=[])
    clean = []
    for row in rows:
        gold = row["gold"]
        if not isinstance(gold, list) or not gold:
            raise ValueError("strict EM requires unchanged gold[0]")
        clean.append({**row, "dialog_id": _dialog_id(row),
                      "strict_score": float(normalized_answer(row["prediction"]) == normalized_answer(gold[0]))})
    images = sorted({str(r["image_id"]) for r in clean})
    scopes = {"all": lambda r: True, "followup": lambda r: int(r["turn_id"]) > 1}
    for t in sorted({int(r["turn_id"]) for r in clean}):
        scopes[f"T{t}"] = lambda r, t=t: int(r["turn_id"]) == t
    quality, latency = [], []
    metrics = ("ttft_ms", "request_e2e_ms", "ssd_read_bytes", "ssd_preads",
               "h2d_kv_bytes", "h2d_ms", "cache_assembly_ms",
               "store_load_ms", "suffix_prefill_ms", "importance_ms",
               "capture_prefix_clone_ms", "vision_qk_capture_bytes",
               "vision_qk_copy_ms", "capture_hook_cpu_ms",
               "importance_peak_extra_gpu_bytes",
               "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
    for method in methods:
        for scope, predicate in scopes.items():
            rs = [r for r in clean if r["method_id"] == method and predicate(r)]
            scores = [r["strict_score"] for r in rs]
            quality.append(dict(method_id=method, scope=scope, requests=len(rs),
                                correct=int(sum(scores)),
                                accuracy=100 * sum(scores) / len(rs) if rs else NOT_MEASURED,
                                accuracy_unit="percent", weighting="request",
                                image_macro_accuracy_percent=statistics(
                                    100 * sum(r["strict_score"] for r in rs if str(r["image_id"]) == iid)
                                    / sum(str(r["image_id"]) == iid for r in rs)
                                    for iid in images if any(str(r["image_id"]) == iid for r in rs))["mean"]))
            item = dict(method_id=method, scope=scope, requests=len(rs), weighting="request")
            for metric in metrics:
                item.update({f"{metric}_{k}": v for k, v in statistics(_field(r, metric) for r in rs).items()})
            macro = []
            for iid in images:
                values = [_field(r, "ttft_ms") for r in rs if str(r["image_id"]) == iid]
                s = statistics(values)
                if s["measured_count"]:
                    macro.append(s["mean"])
            item["ttft_image_macro_mean_ms"] = statistics(macro)["mean"]
            for kind in ("content", "structure", "padding"):
                vals = []
                for r in rs:
                    trace = _trace(r)
                    if trace is not None and (not trace or all("kind" in e for e in trace)):
                        vals.append(sum(_trace_returned(e) for e in trace if e.get("kind") == kind))
                item.update({f"{kind}_read_bytes_{k}": v for k, v in statistics(vals).items()})
            latency.append(item)
    setup, storage = _setup(list(setup_events), methods)
    storage_summary = []
    for method in methods[1:]:
        ss = [r for r in storage if r["method_id"] == method]
        item = dict(method_id=method, images=len(ss), weighting="image_macro")
        for metric in ("file_logical_bytes", "allocated_bytes", "os_write_bytes", "os_write_calls",
                       "content_payload_bytes", "structural_payload_bytes", "padding_bytes",
                       "metadata_bytes", "persistence_ms", "activation_ms"):
            item.update({f"{metric}_{k}": v for k, v in statistics(r[metric] for r in ss).items()})
        measured = [r for r in ss if isinstance(r["N_content"], int) and isinstance(r["stored_k"], int)
                    and r["N_content"] > 0]
        item["content_retention_pooled_percent"] = 100 * sum(r["stored_k"] for r in measured) / sum(r["N_content"] for r in measured) if measured else NOT_MEASURED
        item["content_retention_image_macro_percent"] = statistics(100 * r["stored_k"] / r["N_content"] for r in measured)["mean"]
        storage_summary.append(item)
    by = {(r["dialog_id"], r["method_id"], int(r["turn_id"])): r for r in clean}
    session = []
    # GQA questions are independent, so six requests must not be called a 3-turn session.
    if phase != "gqa_pilot":
        for did in sorted({r["dialog_id"] for r in clean}):
            for method in methods:
                rs = [by.get((did, method, t)) for t in (1, 2, 3)]
                if any(r is None for r in rs):
                    continue
                iid = str(rs[0]["image_id"])
                request = sum(_field(r, "request_e2e_ms") for r in rs)
                if method == methods[0]:
                    s = dict(source_dialog_id=None, build_ms=0., activation_ms=0., setup_ms=0.)
                else:
                    s = setup.get((iid, method))
                source = s is not None and did == s["source_dialog_id"]
                session.append(dict(image_id=iid, dialog_id=did, method_id=method,
                                    request_e2e_sum_ms=request,
                                    actual_build_attribution_ms=s["build_ms"] if source else 0. if s else NOT_MEASURED,
                                    actual_activation_attribution_ms=s["activation_ms"] if source else 0. if s else NOT_MEASURED,
                                    physical_stream_session_ms=request + (s["setup_ms"] if source else 0.) if s else NOT_MEASURED,
                                    derived_standalone_session_ms=request + s["setup_ms"] if s else NOT_MEASURED,
                                    derived_provenance="DERIVED; not independently measured cold-new-image",
                                    non_source_capture_increment=NOT_MEASURED))
    session_summary = []
    for method in methods:
        item = dict(method_id=method, dialogues=sum(r["method_id"] == method for r in session))
        for metric in ("request_e2e_sum_ms", "physical_stream_session_ms", "derived_standalone_session_ms"):
            item.update({f"{metric}_{k}": v for k, v in statistics(r[metric] for r in session if r["method_id"] == method).items()})
        session_summary.append(item)
    attributable = [r for r in session if isinstance(r["actual_build_attribution_ms"], (int, float))]
    attributed = sum(r["actual_build_attribution_ms"] + r["actual_activation_attribution_ms"] for r in attributable)
    physical = sum(s["setup_ms"] for s in setup.values())
    accounting_complete = bool(clean) and phase != "gqa_pilot" and len(setup) == 2 * len(images)
    if accounting_complete and not math.isclose(attributed, physical, abs_tol=1e-7):
        raise ValueError("physical setup/session attribution mismatch")
    pairs, agreement = [], []
    # CI is reported only for complete, audited populations; partial counts remain visible.
    if audit["coverage_complete"] and audit["status"] == "PASS":
        ours = methods[-1]
        for baseline in methods[:-1]:
            for scope, predicate in scopes.items():
                diffs = defaultdict(list)
                agree = Counter()
                scope_images = set()
                for r in clean:
                    if r["method_id"] != baseline or not predicate(r):
                        continue
                    b = by[r["dialog_id"], ours, int(r["turn_id"])]
                    iid = str(r["image_id"])
                    scope_images.add(iid)
                    diffs["accuracy_pp"].append(dict(image_id=iid, value=100 * (b["strict_score"] - r["strict_score"])))
                    for metric in ("ttft_ms", "request_e2e_ms", "ssd_read_bytes"):
                        a, z = _field(b, metric), _field(r, metric)
                        if isinstance(a, (int, float)) and isinstance(z, (int, float)):
                            diffs[metric].append(dict(image_id=iid, value=a-z))
                    agree["requests"] += 1
                    agree["normalized_prediction_equal"] += normalized_answer(b["prediction"]) == normalized_answer(r["prediction"])
                    tok_a, tok_b = _field(b, "generated_token_ids"), _field(r, "generated_token_ids")
                    if tok_a and tok_b:
                        agree["first_token_equal"] += tok_a[0] == tok_b[0]
                        agree["sequence_equal"] += tok_a == tok_b
                    cell = "both_correct" if b["strict_score"] and r["strict_score"] else "baseline_only_correct" if r["strict_score"] else "ours_only_correct" if b["strict_score"] else "both_wrong"
                    agree[cell] += 1
                agreement.append(dict(baseline=baseline, scope=scope, **dict(agree)))
                for metric, values in diffs.items():
                    if len(values) == agree["requests"]:
                        pairs.append(dict(baseline=baseline, scope=scope, metric=metric,
                                          direction="Ours-minus-baseline",
                                          **paired_cluster_ci(values, "value", sorted(scope_images), seed=seed, resamples=resamples)))
            sessions = {(r["dialog_id"], r["method_id"]): r for r in session}
            for metric in ("physical_stream_session_ms", "derived_standalone_session_ms"):
                values = []
                for r in session:
                    if r["method_id"] != baseline:
                        continue
                    b = sessions[r["dialog_id"], ours]
                    if isinstance(r[metric], (int, float)) and isinstance(b[metric], (int, float)):
                        values.append(dict(image_id=r["image_id"], value=b[metric] - r[metric]))
                expected_sessions = sum(r["method_id"] == baseline for r in session)
                if values and len(values) == expected_sessions:
                    pairs.append(dict(baseline=baseline, scope="per_dialogue", metric=metric,
                                      direction="Ours-minus-baseline",
                                      **paired_cluster_ci(values, "value", images, seed=seed, resamples=resamples)))
            stores = {(r["image_id"], r["method_id"]): r for r in storage}
            for metric in ("file_logical_bytes", "allocated_bytes", "os_write_bytes", "persistence_ms", "activation_ms"):
                values = []
                for iid in images:
                    b = stores.get((iid, ours), {}).get(metric)
                    a = 0. if baseline == methods[0] else stores.get((iid, baseline), {}).get(metric)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        values.append(dict(image_id=iid, value=b-a))
                if values and len(values) == len(images):
                    pairs.append(dict(baseline=baseline, scope="image_macro_setup", metric=metric,
                                      direction="Ours-minus-baseline",
                                      **paired_cluster_ci(values, "value", images, seed=seed, resamples=resamples)))
    return dict(status=audit["status"], audit=audit, quality=quality, latency=latency,
                storage=storage, storage_summary=storage_summary, session=session, session_summary=session_summary,
                paired_analysis=pairs, prediction_agreement=agreement,
                session_accounting=dict(status="PASS" if accounting_complete else "NOT_MEASURED",
                                        attributed_setup_ms=attributed if accounting_complete else NOT_MEASURED,
                                        physical_setup_ms=physical if accounting_complete else NOT_MEASURED,
                                        physical_builds=len(setup), activation_events=len(setup),
                                        endpoint="outer writer return; explicit activation receipt"),
                limitations=["Method-level comparisons include generated-history divergence.",
                             "OS I/O bytes do not measure physical NAND traffic.",
                             "Page-cache DONTNEED does not guarantee cold NAND.",
                             "Standalone-equivalent sessions are derived; non-source capture increment is NOT_MEASURED.",
                             "Paired CI spanning zero does not establish equivalence."])


def load_adopted_rows(run: Path | str, phase: str) -> tuple[list[dict], list[dict]]:
    """Read only adopted image attempts, never uncommitted/retry/validation raw."""
    base = Path(run) / phase
    rows, evidence = [], []
    commits = sorted((base / "images").glob("*/COMMITTED.json"))
    for cp in commits:
        commit = json.loads(cp.read_text())
        folder = cp.parent / commit["attempt_id"]
        if not folder.resolve().is_relative_to(cp.parent.resolve()) or folder.name != commit["attempt_id"]:
            raise ValueError("committed attempt escapes its image directory")
        raw = folder / "raw.jsonl"
        if "raw_path" in commit and Path(commit["raw_path"]).resolve() != raw.resolve():
            raise ValueError("committed raw path contradicts owned image attempt")
        digest = file_sha256(raw)
        if digest != commit["raw_sha256"]:
            raise ValueError(f"committed raw hash mismatch: {raw}")
        for name, frozen in commit.get("evidence_sha256", {}).items():
            if file_sha256(folder / name) != frozen:
                raise ValueError(f"committed evidence mismatch: {folder / name}")
        rr = jsonl(raw)
        if len(rr) != commit["requests"]:
            raise ValueError("committed image count mismatch")
        rows.extend(rr)
        evidence.append(dict(path=str(raw), sha256=digest, requests=len(rr),
                             commit=str(cp), commit_sha256=file_sha256(cp)))
    if not commits and (base / "raw.jsonl").exists():
        # A phase stream is accepted only with an explicit completion receipt.
        completion = base / "completion.json"
        if completion.exists() and json.loads(completion.read_text()).get("status") == "PASS":
            raw = base / "raw.jsonl"
            rows = jsonl(raw)
            receipt = json.loads(completion.read_text())
            digest = file_sha256(raw)
            if receipt.get("raw_sha256") != digest:
                raise ValueError("phase completion must bind the full raw hash")
            evidence.append(dict(path=str(raw), sha256=digest, requests=len(rows),
                                 completion=str(completion)))
    return rows, evidence


def load_adopted_setup_events(run: Path | str, phase: str) -> tuple[list[dict], list[dict]]:
    """Load the setup sidecar bound by each adopted image receipt."""
    base = Path(run) / phase
    events, evidence = [], []
    commits = sorted((base / "images").glob("*/COMMITTED.json"))
    for cp in commits:
        commit = json.loads(cp.read_text())
        folder = cp.parent / commit["attempt_id"]
        if not folder.resolve().is_relative_to(cp.parent.resolve()) or folder.name != commit["attempt_id"]:
            raise ValueError("committed setup attempt escapes image ownership")
        path = folder / "setup_events.jsonl"
        if "setup_path" not in commit or Path(commit["setup_path"]).resolve() != path.resolve():
            raise ValueError("committed setup path contradicts owned image attempt")
        digest = file_sha256(path)
        if digest != commit.get("setup_sha256"):
            raise ValueError("committed setup sidecar hash mismatch")
        adopted = jsonl(path)
        if any(e.get("phase", phase) != phase or str(e.get("image_id")) != cp.parent.name for e in adopted):
            raise ValueError("foreign phase/image setup event")
        events.extend(adopted)
        evidence.append(dict(path=str(path), sha256=digest, events=len(adopted),
                             commit=str(cp), commit_sha256=file_sha256(cp)))
    if not commits and (base / "setup_events.jsonl").is_file():
        receipt_path = base / "completion.json"
        if not receipt_path.is_file():
            raise ValueError("phase setup stream lacks completion receipt")
        receipt = json.loads(receipt_path.read_text())
        path = base / "setup_events.jsonl"
        digest = file_sha256(path)
        if receipt.get("status") != "PASS" or receipt.get("setup_sha256") != digest:
            raise ValueError("phase setup completion does not bind sidecar hash")
        events = jsonl(path)
        evidence.append(dict(path=str(path), sha256=digest, events=len(events),
                             completion=str(receipt_path)))
    return events, evidence


def write_csv(path: Path, rows: Sequence[Mapping], fields: Sequence[str]):
    keys = list(dict.fromkeys([*fields, *(k for row in rows for k in row)]))
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})


def write_report(result: Mapping, output: Path | str, *, phase: str,
                 evidence: Sequence[Mapping] = ()) -> Path:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    table_fields = {
        "quality": ("method_id", "scope", "requests", "correct", "accuracy"),
        "latency": ("method_id", "scope", "requests", "ttft_ms_mean", "ttft_ms_p50", "ttft_ms_p95"),
        "storage": ("image_id", "method_id", "file_logical_bytes", "allocated_bytes", "persistence_ms", "activation_ms"),
        "storage_summary": ("method_id", "images", "file_logical_bytes_mean", "persistence_ms_mean", "activation_ms_mean"),
        "session": ("image_id", "dialog_id", "method_id", "physical_stream_session_ms", "derived_standalone_session_ms"),
        "session_summary": ("method_id", "dialogues", "physical_stream_session_ms_mean", "derived_standalone_session_ms_mean"),
        "paired_analysis": ("baseline", "scope", "metric", "point", "ci_low", "ci_high"),
        "prediction_agreement": ("baseline", "scope", "requests", "normalized_prediction_equal"),
    }
    for name, fields in table_fields.items():
        write_csv(out / (name + ".csv"), result.get(name, []), fields)
    for name, value in (("analysis.json", result), ("raw_evidence_index.json", list(evidence))):
        with (out / name).open("x") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    audit = result["audit"]
    lines = [f"# GLM-4.6V-Flash {phase} 독립 결과", "",
             f"상태: **{result['status']}**. 채택 raw 요청 {audit['requests']:,} / 계약 {audit['expected_requests']:,}.",
             "", "정확도는 raw prediction과 원래 gold[0]의 strict normalized exact match를 독립 계산했다.",
             "GPU 정합성 gate는 별도 검증 receipt를 요구한다. 이 보고서가 native correctness를 승인하지 않는다.", ""]
    if not audit["requests"]:
        lines += ["GLM 생성·latency·SSD 요청 읽기·setup·session·paired CI 실측이 없다. 모든 미측정 결과는 NOT_MEASURED다.", ""]
    elif result["status"] != "PASS":
        lines += ["불완전하거나 거부된 evidence의 결과다. 완전한 pilot/main 완료로 해석하지 않는다. paired CI는 생성하지 않았다.", ""]
    else:
        lines += ["| 방법 | 요청 | 정확도 % | Followup TTFT mean/p50/p95 ms | 실제 KV read MB/request |",
                  "|---|---:|---:|---|---:|"]
        for q in result["quality"]:
            if q["scope"] != "all":
                continue
            lat = next((x for x in result["latency"] if x["method_id"] == q["method_id"] and x["scope"] == "followup"), {})
            times = "/".join(f"{lat['ttft_ms_'+s]:.3f}" if isinstance(lat.get('ttft_ms_'+s), (int, float)) else NOT_MEASURED for s in ("mean", "p50", "p95"))
            read = lat.get("ssd_read_bytes_mean")
            read = f"{read / 1e6:.6f}" if isinstance(read, (int, float)) else NOT_MEASURED
            lines.append(f"| {q['method_id']} | {q['requests']} | {q['accuracy']:.3f} | {times} | {read} |")
        lines += ["", "방법 비교는 각 방법이 생성한 자기 history의 차이를 포함한다. MB=10^6 bytes, GiB=2^30 bytes, ms, 정확도 차이는 percentage points다.", ""]
    lines += ["actual shared-image session은 이미지별 실제 build와 activation을 한 번만 귀속한다.",
              "standalone-equivalent는 이미지 setup을 대화마다 다시 부과한 DERIVED 값이다. 비-source capture 추가 비용은 NOT_MEASURED다.",
              "", "paired_analysis.csv는 seed1234, 이미지 단위 paired cluster bootstrap 10,000회다.",
              "같은 이미지의 방법/대화/turn을 함께 resample하고 pooled sum/count로 요청 가중치를 보존한다.",
              "CI가 0을 포함해도 동등성을 주장하지 않는다. OS read/write를 물리 NAND I/O로 부르지 않는다.",
              "", "전체 source·membership·history·selection·I/O 감사와 count는 analysis.json, raw hash는 raw_evidence_index.json에 있다."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return out

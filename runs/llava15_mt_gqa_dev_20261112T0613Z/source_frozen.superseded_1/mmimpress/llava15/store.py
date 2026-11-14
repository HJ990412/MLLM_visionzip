"""Ratio-general PackedPrefixStore for LLaVA-1.5-7B (``llava15_prefixstore_ratio_packed_v1``).

One image == one ``kv_packed.bin``::

    [L0 K rows 0..k_0) [L0 V rows 0..k_0) [L1 K] [L1 V] ... [L31 V]

FP16, token-major ``(rows, num_heads, head_dim)`` per block, in the global
image-only VisionZip order (CLIP penultimate-layer CLS->patch attention summed
over heads, stable descending sort).  Every block is produced by the same
operations as ``mmimpress.packed_store.write_packed_store`` (materialise,
permute, ``index_select`` by the permutation, slice ``[:k_l]``, ``tobytes``).

Differences from the LLaVA-NeXT packed store, all forced by the model:

* LLaVA-1.5 has exactly 576 visual tokens and **no separator**; the trailing
  separator sidecar is 0 bytes and the structural tail is empty;
* the total retention ratio is a property of the frozen budget document
  (1.0 / 0.5 / 0.25), not the constant 0.25 of ``packed_store`` /
  ``packed_serve`` / ``prefix25``;
* every ``mmimpress.layer_budget`` call passes ``num_layers``, ``total_ratio``,
  ``r_min`` and ``r_max`` explicitly; the clip range of this port is the
  contract constant ``[0.0, 1.0]`` and any other value fails closed.

Contract: docs/llava15_mt_gqa_contract.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from mmimpress import layer_budget as LB
from mmimpress.cvpr25 import permutation_sha256, visionzip_repack_order
from mmimpress.llava15 import MODEL_ID, N_VISUAL, NUM_LAYERS, R_MAX, R_MIN
from mmimpress.model import cache_layers
from mmimpress.packed_store import PACKED_NAME, packed_layout
from mmimpress.store import META_NAME

STORE_FORMAT = "llava15_prefixstore_ratio_packed_v1"
META_FORMAT = "llava15_fp16_importance_prefix_ratio_packed_v1"
STORAGE_POLICY = "prefix_ratio_packed"
BUDGET_SCHEMA = "llava15_layer_budget_v1"
ROUNDING = "hamilton_largest_remainder_total_ceil_ratio_times_original_content"
LAYOUT_POLICY = "visionzip_image_only"
SALIENCY_SOURCE = "turn1_vision_penultimate_cls_to_patch_attention_head_sum"


# ------------------------------------------------------------------ budget
def make_budget(alloc_id: str, ratios, total_ratio: float, source: Mapping[str, Any]) -> dict:
    """Frozen per-arm budget document.  Clip range is the contract constant."""
    r = LB.validate_ratios(ratios, num_layers=NUM_LAYERS, total_ratio=float(total_ratio), r_min=R_MIN, r_max=R_MAX)
    return {"schema": BUDGET_SCHEMA, "policy": LB.POLICY, "alloc_id": str(alloc_id), "model": MODEL_ID,
            "num_layers": NUM_LAYERS, "total_ratio": float(total_ratio), "r_min": R_MIN, "r_max": R_MAX,
            "layer_keep_ratios": r, "layer_budget_sha256": LB.budget_sha256(r), "source": dict(source),
            "rounding": "hamilton_largest_remainder; floor then +1 by descending fractional remainder "
                        "(ties: lower layer first); cap N; sum == L*ceil(total_ratio*N)"}


def validate_budget(doc: Mapping[str, Any]) -> list[float]:
    """Fail-closed check of a budget document; returns the ratio list."""
    if not isinstance(doc, Mapping) or doc.get("schema") != BUDGET_SCHEMA or doc.get("policy") != LB.POLICY:
        raise ValueError("not a llava15_layer_budget_v1 document")
    for key in ("num_layers", "total_ratio", "r_min", "r_max", "layer_keep_ratios", "layer_budget_sha256", "alloc_id"):
        if key not in doc:
            raise ValueError(f"budget document lacks {key}; module defaults are never used")
    if int(doc["num_layers"]) != NUM_LAYERS:
        raise ValueError("budget document is not for 32 layers")
    if float(doc["r_min"]) != R_MIN or float(doc["r_max"]) != R_MAX:
        raise ValueError(f"LLaVA-1.5 budgets use the contract clip range [{R_MIN}, {R_MAX}] only")
    total = float(doc["total_ratio"])
    if not (0.0 < total <= 1.0):
        raise ValueError("total_ratio outside (0, 1]")
    r = LB.validate_ratios(doc["layer_keep_ratios"], num_layers=NUM_LAYERS, total_ratio=total, r_min=R_MIN, r_max=R_MAX)
    if doc["layer_budget_sha256"] != LB.budget_sha256(r):
        raise ValueError("layer budget SHA256 does not match its ratios")
    if not isinstance(doc["alloc_id"], str) or not doc["alloc_id"]:
        raise ValueError("budget document lacks an alloc_id")
    return r


def load_budget(path) -> dict:
    doc = json.loads(Path(path).read_text())
    doc["layer_keep_ratios"] = validate_budget(doc)
    return doc


def keep_counts(doc: Mapping[str, Any], n_content: int) -> list[int]:
    """Per-layer content rows ``k_l`` (Hamilton apportionment of ``32*ceil(r*N)``)."""
    r = validate_budget(doc)
    counts = LB.layer_keep_counts(r, int(n_content), num_layers=NUM_LAYERS, total_ratio=float(doc["total_ratio"]),
                                  r_min=R_MIN, r_max=R_MAX)
    if sum(counts) != row_budget(doc, n_content):
        raise ValueError("layer keep counts do not sum to the row budget")
    return counts


def row_budget(doc: Mapping[str, Any], n_content: int) -> int:
    return LB.total_row_budget(int(n_content), num_layers=NUM_LAYERS, total_ratio=float(doc["total_ratio"]))


def uniform_equivalent(doc: Mapping[str, Any], n_content: int) -> int:
    return row_budget(doc, n_content) // NUM_LAYERS


# ------------------------------------------------------------------ writer
def write_llava15_packed_store(out_dir: Path, layers, v_start: int, v_num: int, prefix_input_ids, order, *,
                               budget: Mapping[str, Any], extra: Mapping[str, Any] | None = None,
                               timing_out: dict | None = None) -> dict:
    """Write one packed LLaVA-1.5 image prefix; returns the meta dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = validate_budget(budget)
    td = torch.float16
    v_start, v_num = int(v_start), int(v_num)
    prefix_len = v_start + v_num
    L = len(layers)
    if L != NUM_LAYERS or v_num != N_VISUAL:
        raise ValueError("LLaVA-1.5 packed store requires 32 layers and 576 contiguous visual tokens")
    H, hd = layers[0][0].shape[1], layers[0][0].shape[3]
    assert layers[0][0].shape[2] >= prefix_len, layers[0][0].shape
    n_content = v_num                                   # no separator, no padding
    order = [int(i) for i in order]
    if len(order) != v_num or sorted(order) != list(range(v_num)):
        raise ValueError("order is not a permutation of the 576 visual tokens")
    rows_per_layer = keep_counts(budget, n_content)
    assert len(rows_per_layer) == L and all(0 <= k <= n_content for k in rows_per_layer)
    perm = torch.tensor(order, dtype=torch.long)
    inverse = [0] * v_num
    for stored, original in enumerate(order):
        inverse[original] = stored
    row_bytes = H * hd * 2
    table, sep_offset, sep_bytes, total_bytes = packed_layout(rows_per_layer, row_bytes, L, 0)
    assert sep_bytes == 0 and sep_offset == total_bytes

    materialize_s = repack_s = write_s = serialize_s = hash_s = 0.0
    t0 = time.perf_counter()
    sys_k = torch.stack([k[0, :, :v_start].to(td).cpu() for k, _ in layers])
    sys_v = torch.stack([v[0, :, :v_start].to(td).cpu() for _, v in layers])
    materialize_s += time.perf_counter() - t0
    t0 = time.perf_counter()
    torch.save({"k": sys_k, "v": sys_v}, out_dir / "sys_kv.pt")
    write_s += time.perf_counter() - t0

    digest = hashlib.sha256()
    written = 0
    path = out_dir / PACKED_NAME
    with path.open("wb") as f:
        for li, (k, v) in enumerate(layers):
            for kind, t in (("k", k), ("v", v)):
                t0 = time.perf_counter()
                block = t[0, :, v_start:prefix_len].permute(1, 0, 2).to(td).contiguous().cpu()
                assert block.shape == (v_num, H, hd), block.shape
                materialize_s += time.perf_counter() - t0
                t0 = time.perf_counter()
                block = block.index_select(0, perm).contiguous()
                repack_s += time.perf_counter() - t0
                t0 = time.perf_counter()
                buf = block[:rows_per_layer[li]].numpy().tobytes()
                serialize_s += time.perf_counter() - t0
                assert written == table[li]["k_offset" if kind == "k" else "v_offset"], (li, kind, written)
                t0 = time.perf_counter()
                f.write(buf)
                write_s += time.perf_counter() - t0
                t0 = time.perf_counter()
                digest.update(buf)
                hash_s += time.perf_counter() - t0
                written += len(buf)
        f.flush()
    assert written == total_bytes == path.stat().st_size
    max_rows = max(rows_per_layer)
    timing = {"kv_materialize_ms": materialize_s * 1e3, "kv_repack_ms": repack_s * 1e3, "ssd_write_ms": write_s * 1e3,
              "prefix_slice_serialization_ms": serialize_s * 1e3, "packed_sha256_ms": hash_s * 1e3}
    meta = {
        "v_token_start": v_start, "v_token_num": v_num, "prefix_len": prefix_len,
        "num_layers": L, "num_heads": H, "head_dim": hd, "probe_heads": 0, "dtype": "float16",
        "newline_idx": [], "newline_stored": [], "n_spatial": n_content,
        "prefix_input_ids": list(prefix_input_ids),
        "layout": "packed: one kv_packed.bin = [L0 K rows 0..k_0)[L0 V][L1 K][L1 V]...[L31 V]; no separator sidecar; "
                  "each block token-major (rows, num_heads, head_dim) fp16 in the global importance order",
        "bytes_visual_kv": total_bytes, "bytes_probe_sidecar": 0, "bytes_separator_sidecar": 0,
        "order": list(order), "order_is_per_layer": False, "reordered": True,
        "format": META_FORMAT, "storage_policy": STORAGE_POLICY, "store_format": STORE_FORMAT, "packed_file": PACKED_NAME,
        "budget_policy": LB.POLICY, "budget_schema": BUDGET_SCHEMA,
        "original_content_count": n_content,
        "uniform_equivalent_content_count": uniform_equivalent(budget, n_content),
        "stored_content_count_per_layer": list(rows_per_layer),
        "stored_content_count_total": int(sum(rows_per_layer)),
        "payload_rows_per_layer": list(rows_per_layer),
        "retention_ratio": float(budget["total_ratio"]),
        "rounding_policy": ROUNDING,
        "layer_budget_id": str(budget["alloc_id"]),
        "layer_keep_ratios": [float(x) for x in ratios],
        "layer_keep_counts": list(rows_per_layer),
        "layer_budget_sha256": str(budget["layer_budget_sha256"]),
        "layer_budget_total_ratio": float(budget["total_ratio"]),
        "layer_budget_r_min": float(budget["r_min"]), "layer_budget_r_max": float(budget["r_max"]),
        "layer_budget_num_layers": int(budget["num_layers"]),
        "stored_row_to_original": list(order)[:max_rows],
        "full_importance_permutation": list(order),
        "order_semantics": "full virtual GPU permutation; layer l stores stored_row_to_original[:layer_keep_counts[l]]",
        "padding_rows": 0, "packed_row_bytes": row_bytes, "packed_layout": table,
        "packed_sep_offset": sep_offset, "packed_sep_bytes": 0,
        "packed_total_bytes": total_bytes, "packed_sha256": digest.hexdigest(),
        "packed_writer_timing_ms": timing,
    }
    if extra:
        reserved = set(meta) & set(extra)
        if reserved:
            raise ValueError("extra metadata cannot override writer fields: " + ", ".join(sorted(reserved)))
        meta.update(extra)
    t0 = time.perf_counter()
    with open(out_dir / META_NAME, "w") as f:
        json.dump(meta, f, indent=1)
    timing["ssd_write_ms"] += (time.perf_counter() - t0) * 1e3
    if timing_out is not None:
        timing_out.update(timing)
    return meta


# --------------------------------------------------------------- validator
def meta_budget(m: Mapping[str, Any]) -> dict:
    """Budget document reconstructed from store metadata only (no defaults)."""
    for key in ("layer_budget_total_ratio", "layer_budget_r_min", "layer_budget_r_max", "layer_budget_num_layers",
                "layer_keep_ratios", "layer_budget_sha256", "layer_budget_id"):
        if key not in m:
            raise ValueError(f"store metadata lacks {key}")
    return {"schema": BUDGET_SCHEMA, "policy": m.get("budget_policy"), "alloc_id": m["layer_budget_id"],
            "num_layers": m["layer_budget_num_layers"], "total_ratio": m["layer_budget_total_ratio"],
            "r_min": m["layer_budget_r_min"], "r_max": m["layer_budget_r_max"],
            "layer_keep_ratios": m["layer_keep_ratios"], "layer_budget_sha256": m["layer_budget_sha256"]}


def validate_llava15_packed_meta(m: Mapping[str, Any], expected_budget: Mapping[str, Any] | None = None) -> None:
    """Fail-closed check of geometry, vector, counts, mapping and byte table."""
    if (m.get("format") != META_FORMAT or m.get("storage_policy") != STORAGE_POLICY
            or m.get("store_format") != STORE_FORMAT or m.get("packed_file") != PACKED_NAME
            or m.get("probe_heads") != 0 or m.get("bytes_probe_sidecar") != 0
            or m.get("dtype") != "float16" or "payload_rows" in m
            or m.get("budget_schema") != BUDGET_SCHEMA or m.get("budget_policy") != LB.POLICY):
        raise ValueError("invalid LLaVA-1.5 packed store format")
    vn, L = int(m["v_token_num"]), int(m["num_layers"])
    if (vn != N_VISUAL or L != NUM_LAYERS or m.get("newline_idx") != [] or m.get("newline_stored") != []
            or int(m.get("n_spatial", -1)) != vn or m.get("original_content_count") != vn
            or int(m["prefix_len"]) != int(m["v_token_start"]) + vn
            or int(m["num_heads"]) != 32 or int(m["head_dim"]) != 128):
        raise ValueError("invalid LLaVA-1.5 visual geometry (576 contiguous tokens, no separator)")
    try:
        doc = meta_budget(m)
        ratios = validate_budget(doc)
        counts = keep_counts(doc, vn)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid layer budget in store metadata: {exc}") from exc
    total_ratio = float(doc["total_ratio"])
    if (m.get("layer_keep_counts") != counts or m.get("payload_rows_per_layer") != counts
            or m.get("stored_content_count_per_layer") != counts
            or m.get("stored_content_count_total") != sum(counts)
            or sum(counts) != L * math.ceil(total_ratio * vn)
            or m.get("uniform_equivalent_content_count") != math.ceil(total_ratio * vn)
            or m.get("retention_ratio") != total_ratio or m.get("rounding_policy") != ROUNDING
            or m.get("padding_rows") != 0):
        raise ValueError("layer keep counts disagree with the ratio vector")
    order = m.get("order")
    if (not isinstance(order, list) or sorted(order) != list(range(vn))
            or m.get("full_importance_permutation") != order or m.get("order_is_per_layer") is not False
            or m.get("stored_row_to_original") != order[:max(counts)]):
        raise ValueError("invalid importance permutation or row mapping")
    row_bytes = int(m["num_heads"]) * int(m["head_dim"]) * 2
    table, sep_offset, sep_bytes, total = packed_layout(counts, row_bytes, L, 0)
    if (m.get("packed_row_bytes") != row_bytes or m.get("packed_layout") != table
            or m.get("packed_sep_offset") != sep_offset or m.get("packed_sep_bytes") != 0
            or m.get("packed_total_bytes") != total or m.get("bytes_separator_sidecar") != 0
            or m.get("bytes_visual_kv") != total
            or not isinstance(m.get("packed_sha256"), str) or len(m["packed_sha256"]) != 64):
        raise ValueError("packed byte table disagrees with the row counts")
    if expected_budget is not None:
        exp = validate_budget(expected_budget)
        if (m["layer_budget_sha256"] != expected_budget["layer_budget_sha256"] or ratios != exp
                or m["layer_budget_id"] != expected_budget["alloc_id"]
                or total_ratio != float(expected_budget["total_ratio"])):
            raise ValueError("store was built for a different budget vector")


# ------------------------------------------------------------- persistence
CAPTURE_REQUIREMENTS = {
    "capture_saliency": True, "vision_call_count": 1, "saliency_call_count": 1,
    "extra_vision_forward_calls": 0, "extra_saliency_forward_ms": 0.0, "saliency_layer_from_end": 2,
    "vision_attention_backend": "eager", "global_vision_output_attentions": False,
    "global_model_output_attentions": False,
}


def token_scores_from_capture(per_crop_scores) -> torch.Tensor:
    """(1, 576) CLS->patch saliency of the single crop -> (576,) token scores.

    LLaVA-1.5 feeds the 576 patches of one crop in raster order, so the
    visual-token coordinate is the patch index itself (no AnyRes mapping).
    """
    scores = torch.as_tensor(per_crop_scores).float().cpu()
    if tuple(scores.shape) != (1, N_VISUAL):
        raise ValueError(f"expected CLS saliency of shape (1, {N_VISUAL}), got {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise ValueError("CLS saliency contains NaN/Inf")
    return scores[0].clone()


@torch.no_grad()
def persist_captured_llava15_prefix(runner, captured_past_key_values, expanded_input_ids, per_crop_scores, out_dir, *,
                                    budget: Mapping[str, Any], image_id, capture_stats, image_input_sha256=None,
                                    extra_metadata: Mapping[str, Any] | None = None) -> dict:
    """Turn-1 piggyback persistence (LLaVA-1.5 counterpart of
    ``mmimpress.piggyback.persist_captured_visual_prefix`` + packed writer).

    Never calls the model or the vision tower.  Same capture-provenance
    requirements, staging directory, fsync order, ``RENAME_NOREPLACE``
    publication and receipt layout as the LLaVA-NeXT flow.
    """
    from mmimpress import piggyback as pb
    persist_t0 = time.perf_counter()
    destination = Path(out_dir)
    if destination.name in ("", ".", ".."):
        raise ValueError(f"unsafe output directory: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError("refusing to overwrite store", str(destination))
    if isinstance(capture_stats, pb.VisionForwardCapture):
        assert capture_stats.runner is runner, "capture object belongs to a different runner/model"
        assert torch.equal(torch.as_tensor(per_crop_scores).float().cpu(), capture_stats.result_cpu()), \
            "per_crop_scores differ from the completed capture object"
        capture_document = capture_stats.stats()
    else:
        assert capture_stats is not None, "capture_stats are required to prove same-request piggyback provenance"
        capture_document = dict(capture_stats)
    for key, expected in CAPTURE_REQUIREMENTS.items():
        assert capture_document.get(key) == expected, f"invalid Turn-1 capture provenance {key}: {capture_document.get(key)!r}"
    assert capture_document.get("vision_ms") is not None
    assert int(capture_document.get("saliency_layer_index", -1)) == int(capture_document.get("vision_num_layers", -1)) - 2

    ids = expanded_input_ids
    if ids.ndim == 2:
        assert ids.shape[0] == 1, ids.shape
        ids = ids[0]
    v_start, v_num = runner.visual_span(ids)
    base, separators = runner.flat_layout(v_num)
    assert separators == []
    prefix_len = int(v_start + v_num)
    layers = cache_layers(captured_past_key_values)
    assert len(layers) == NUM_LAYERS
    for li, (key, value) in enumerate(layers):
        assert key.ndim == value.ndim == 4 and key.shape == value.shape, (li, key.shape, value.shape)
        assert key.shape[0] == 1 and key.shape[2] >= prefix_len, (li, key.shape, prefix_len)

    t0 = time.perf_counter()
    token_scores = token_scores_from_capture(per_crop_scores)
    token_mapping_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    order = visionzip_repack_order(token_scores, [])
    inverse = [0] * v_num
    for stored, original in enumerate(order):
        inverse[original] = stored
    permutation_ms = (time.perf_counter() - t0) * 1e3

    invariant = {
        "image_id": str(image_id), "model": MODEL_ID, "base_grid": int(base), "hires_grid": None,
        "physical_layout": LAYOUT_POLICY, "layout_method": LAYOUT_POLICY,
        "layout_source": "turn1_normal_inference_piggyback", "visual_kv_source": "turn1_captured_past_key_values",
        "saliency_source": SALIENCY_SOURCE, "turn1_normal_inference": True,
        "separate_vision_forward": False, "separate_prefix_forward": False, "composed_from_store": False,
        "layout_uses_dataset_question": False, "llm_used_for_layout_scoring": False, "calibration_questions": 0,
        "turn1_question_present_in_source_forward": True, "visual_prefix_causally_precedes_question": True,
        "global_order_all_layers": True, "separator_policy": "none_llava15_has_no_separator", "separator_tail": False,
        "probe_heads_required_for_serving": 0,
        "permutation_sha256": permutation_sha256(order), "inverse_permutation_sha256": permutation_sha256(inverse),
        "turn1_capture": capture_document, "capture_provenance_validated": True,
    }
    if image_input_sha256 is not None:
        invariant["image_input_sha256"] = str(image_input_sha256)
    if extra_metadata:
        conflicts = set(extra_metadata) & set(invariant)
        if conflicts:
            raise ValueError("extra_metadata cannot override invariants: " + ", ".join(sorted(conflicts)))
        invariant.update(dict(extra_metadata))

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    published = False
    try:
        writer_timing: dict[str, float] = {}
        meta = write_llava15_packed_store(staging, layers, int(v_start), int(v_num), ids[:prefix_len].detach().cpu().tolist(),
                                          order, budget=budget, extra=invariant, timing_out=writer_timing)
        layout_artifact = {
            "schema_version": 2, "image_id": str(image_id), "importance_source": SALIENCY_SOURCE,
            "capture_source": "same_turn1_normal_inference", "token_score_original": token_scores,
            "stored_to_original": torch.tensor(order, dtype=torch.int32),
            "original_to_stored": torch.tensor(inverse, dtype=torch.int32),
            "newline_original": torch.tensor([], dtype=torch.int32),
            "layout_uses_dataset_question": False, "llm_used_for_layout_scoring": False, "calibration_questions": 0,
            "separate_vision_forward": False, "separate_prefix_forward": False, "image_input_sha256": image_input_sha256,
        }
        t0 = time.perf_counter()
        torch.save(layout_artifact, staging / "visionzip_layout.pt")
        layout_write_ms = (time.perf_counter() - t0) * 1e3
        file_sizes = pb._store_file_sizes(staging)
        file_fsync_ms, directory_fsync_ms, synced_files, synced_dirs = pb._fsync_staging_tree(staging)
        t0 = time.perf_counter()
        pb._rename_noreplace(staging, destination)
        atomic_rename_ms = (time.perf_counter() - t0) * 1e3
        published = True
        t0 = time.perf_counter()
        pb._fsync_directory(parent)
        parent_fsync_ms = (time.perf_counter() - t0) * 1e3
        durability_ms = file_fsync_ms + directory_fsync_ms + atomic_rename_ms + parent_fsync_ms
        persist_ms = (time.perf_counter() - persist_t0) * 1e3        # store-ready boundary

        t0 = time.perf_counter()
        sample_hash = pb._sampled_store_sha256(destination, file_sizes)
        small_hashes = {rel: pb.sha256_file(destination / rel) for rel in ("meta.json", "visionzip_layout.pt")}
        hash_ms = (time.perf_counter() - t0) * 1e3
        total = int(sum(file_sizes.values()))
        known = (int(file_sizes.get("sys_kv.pt", 0)) + int(file_sizes.get("visionzip_layout.pt", 0))
                 + int(file_sizes.get("meta.json", 0)) + int(meta["bytes_visual_kv"]))
        timing_ms = {
            "prefix_slice_serialization_ms": writer_timing["prefix_slice_serialization_ms"],
            "token_mapping_ms": float(token_mapping_ms), "permutation_ms": float(permutation_ms),
            "kv_materialize_ms": float(writer_timing["kv_materialize_ms"]), "kv_repack_ms": float(writer_timing["kv_repack_ms"]),
            "repack_ms": float(writer_timing["kv_materialize_ms"]) + float(writer_timing["kv_repack_ms"]),
            "store_write_ms": float(writer_timing["ssd_write_ms"]), "layout_write_ms": float(layout_write_ms),
            "ssd_write_ms": float(writer_timing["ssd_write_ms"]) + float(layout_write_ms),
            "packed_sha256_ms": float(writer_timing["packed_sha256_ms"]),
            "hash_ms": float(hash_ms), "integrity_hash_ms": float(hash_ms),
            "file_fsync_ms": float(file_fsync_ms), "directory_fsync_ms": float(directory_fsync_ms),
            "atomic_rename_ms": float(atomic_rename_ms), "parent_fsync_ms": float(parent_fsync_ms),
            "durability_ms": float(durability_ms), "persist_ms": float(persist_ms),
            "helper_total_ms": (time.perf_counter() - persist_t0) * 1e3,
        }
        return {
            "store_dir": str(destination.resolve()), "image_id": str(image_id), "meta": meta, "timing_ms": timing_ms,
            "bytes": {"visual_kv": int(meta["bytes_visual_kv"]), "probe_sidecar": 0, "separator_sidecar": 0,
                      "system_prefix": int(file_sizes.get("sys_kv.pt", 0)),
                      "layout_metadata": int(file_sizes.get("visionzip_layout.pt", 0)),
                      "meta": int(file_sizes.get("meta.json", 0)), "other": total - known, "total": total},
            "file_sizes": file_sizes,
            "hashes": {"files_sha256": small_hashes, "prefix_kv_sample_sha256": sample_hash,
                       "packed_sha256": meta["packed_sha256"], "permutation_sha256": meta["permutation_sha256"],
                       "inverse_permutation_sha256": meta["inverse_permutation_sha256"]},
            "durability": {"same_filesystem_staging": True, "atomic_no_clobber": True, "files_fsynced": int(synced_files),
                           "directories_fsynced_before_rename": int(synced_dirs), "parent_fsynced_after_rename": True},
        }
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)

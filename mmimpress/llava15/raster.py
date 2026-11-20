"""Full raster visual-KV store for LLaVA-1.5-7B (FullLoad / IMPRESS arms).

The LLaVA-NeXT FullLoad and IMPRESS arms serve the store written by
``mmimpress.piggyback.persist_captured_raster_prefix``: every visual K/V row in
the model's own order, token-major, plus the probe-key sidecar (the first
``probe_heads`` heads of every key).  This is the LLaVA-1.5 counterpart.  The
bytes are produced by the same ``mmimpress.store.write_image_store`` call
(``stored_to_original=None``, ``separator_sidecar=True``), so the existing
``ImageContext`` / ``ChunkReader`` / ``LayerSelector`` serve it unchanged.

Differences from the LLaVA-NeXT writer, all forced by the model:

* 576 contiguous visual tokens, no AnyRes grid, **no separator**: the
  separator sidecar ``sep_kv.bin`` is an empty file (``ImageContext`` requires
  the file to exist with exactly ``2 * L * n_sep * H * hd * 2`` = 0 bytes);
* no ``image_sizes``;
* the layer-0 visual hidden states (``v_hidden.pt``) are not stored.  They
  feed only the SparseVLM rater selection, which the H2O importance metric of
  the IMPRESS arm does not use (``ImageContext(require_v_hidden=False)``).

Publication follows the same staging directory -> fsync -> ``RENAME_NOREPLACE``
protocol as the packed LLaVA-1.5 store.

Contract: docs/llava15_mtvqa_impress_contract.md.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from mmimpress.cvpr25 import permutation_sha256
from mmimpress.llava15 import MODEL_ID, N_VISUAL, NUM_LAYERS
from mmimpress.model import cache_layers
from mmimpress.store import write_image_store

LAYOUT = "raster"
SEPARATOR_POLICY = "none_llava15_has_no_separator"
_WRITER_RESERVED = {
    "v_token_start", "v_token_num", "prefix_len", "num_layers", "num_heads", "head_dim", "probe_heads", "dtype",
    "chunk_size", "n_chunks_per_layer", "newline_idx", "newline_stored", "n_spatial", "prefix_input_ids", "layout",
    "order", "bytes_visual_kv", "bytes_probe_sidecar", "bytes_separator_sidecar",
}


def validate_llava15_raster_meta(m: Mapping[str, Any], probe_heads: int | None = None) -> None:
    """Fail-closed geometry check of a (possibly importance-reordered) full store."""
    vn, L = int(m["v_token_num"]), int(m["num_layers"])
    if (vn != N_VISUAL or L != NUM_LAYERS or m.get("newline_idx") != [] or int(m.get("n_spatial", -1)) != vn
            or int(m["prefix_len"]) != int(m["v_token_start"]) + vn or int(m["num_heads"]) != 32
            or int(m["head_dim"]) != 128 or m.get("dtype") != "float16"):
        raise ValueError("invalid LLaVA-1.5 full-store geometry (576 contiguous tokens, no separator)")
    if m.get("storage_policy", "full") != "full" or m.get("separator_policy") != SEPARATOR_POLICY:
        raise ValueError("not a full LLaVA-1.5 visual-KV store")
    width = int(m["num_heads"]) * int(m["head_dim"]) * 2
    if int(m["bytes_visual_kv"]) != 2 * L * vn * width or int(m["bytes_separator_sidecar"]) != 0:
        raise ValueError("full store byte totals disagree with the geometry")
    if probe_heads is not None and int(m["probe_heads"]) != int(probe_heads):
        raise ValueError("probe sidecar width disagrees with the request")
    if int(m["bytes_probe_sidecar"]) != L * vn * int(m["probe_heads"]) * int(m["head_dim"]) * 2:
        raise ValueError("probe sidecar byte total disagrees with the geometry")


@torch.no_grad()
def persist_captured_llava15_raster_prefix(runner, captured_past_key_values, expanded_input_ids, out_dir, *,
                                           image_id, probe_heads: int, chunk_size: int = 64,
                                           image_input_sha256: str | None = None,
                                           extra_metadata: Mapping[str, Any] | None = None) -> dict:
    """Write the whole visual KV of one Turn-1 capture in raster order.

    Never calls the model or the vision tower.  Returns the receipt layout of
    the packed LLaVA-1.5 writer (``store_dir``, ``meta``, ``timing_ms``,
    ``bytes``, ``file_sizes``, ``hashes``, ``durability``).
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

    ids = expanded_input_ids
    if ids.ndim == 2:
        assert ids.shape[0] == 1, ids.shape
        ids = ids[0]
    v_start, v_num = runner.visual_span(ids)
    base, separators = runner.flat_layout(v_num)
    assert separators == [] and int(v_num) == N_VISUAL
    prefix_len = int(v_start + v_num)
    layers = cache_layers(captured_past_key_values)
    if len(layers) != NUM_LAYERS:
        raise ValueError("LLaVA-1.5 full store requires 32 layers")
    for li, (key, value) in enumerate(layers):
        assert key.ndim == value.ndim == 4 and key.shape == value.shape, (li, key.shape, value.shape)
        assert key.shape[0] == 1 and key.shape[2] >= prefix_len, (li, key.shape, prefix_len)
    heads = int(layers[0][0].shape[1])
    if not 0 < int(probe_heads) <= heads:
        raise ValueError(f"probe_heads must be in [1,{heads}]")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")

    identity_sha = permutation_sha256(list(range(int(v_num))))
    invariant = {
        "image_id": str(image_id), "model": MODEL_ID, "base_grid": int(base), "hires_grid": None,
        "physical_layout": LAYOUT, "layout_method": LAYOUT,
        "layout_source": "turn1_normal_inference_piggyback", "visual_kv_source": "turn1_captured_past_key_values",
        "visual_hidden_source": None, "turn1_normal_inference": True,
        "separate_vision_forward": False, "separate_prefix_forward": False, "composed_from_store": False,
        "layout_uses_dataset_question": False, "layout_uses_generated_answer": False,
        "llm_used_for_layout_scoring": False, "calibration_questions": 0, "future_questions_used": 0,
        "turn1_question_present_in_source_forward": True, "visual_prefix_causally_precedes_question": True,
        "global_order_all_layers": True, "separator_policy": SEPARATOR_POLICY, "separator_tail": False,
        "reordered": False, "order_is_per_layer": False, "identity_permutation_sha256": identity_sha,
        "probe_heads_required_for_serving": int(probe_heads), "qa_select_compatible": False,
        "storage_policy": "full", "store_format": "llava15_full_raster_v1",
    }
    if image_input_sha256 is not None:
        invariant["image_input_sha256"] = str(image_input_sha256)
    if extra_metadata:
        conflicts = (set(extra_metadata) & set(invariant)) | (set(extra_metadata) & _WRITER_RESERVED)
        if conflicts:
            raise ValueError("extra_metadata cannot override invariants: " + ", ".join(sorted(conflicts)))
        invariant.update(dict(extra_metadata))

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    published = False
    try:
        writer_timing: dict[str, float] = {}
        meta = write_image_store(staging, layers, int(v_start), int(v_num), ids[:prefix_len].detach().cpu().tolist(), [],
                                 extra=invariant, chunk_size=int(chunk_size), probe_heads=int(probe_heads),
                                 stored_to_original=None, separator_sidecar=True, timing_out=writer_timing)
        assert meta["reordered"] is False and "order" not in meta and meta["newline_idx"] == []
        assert (staging / "sep_kv.bin").stat().st_size == 0
        validate_llava15_raster_meta(meta, probe_heads)
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
        small_hashes = {"meta.json": pb.sha256_file(destination / "meta.json")}
        hash_ms = (time.perf_counter() - t0) * 1e3
        total = int(sum(file_sizes.values()))
        known = (int(meta["bytes_visual_kv"]) + int(meta["bytes_probe_sidecar"]) + int(file_sizes.get("sys_kv.pt", 0))
                 + int(file_sizes.get("meta.json", 0)))
        timing_ms = {
            "kv_materialize_ms": float(writer_timing["kv_materialize_ms"]), "kv_repack_ms": float(writer_timing["kv_repack_ms"]),
            "repack_ms": 0.0, "prefix_slice_serialization_ms": float(writer_timing["prefix_slice_serialization_ms"]),
            "store_write_ms": float(writer_timing["ssd_write_ms"]), "ssd_write_ms": float(writer_timing["ssd_write_ms"]),
            "hash_ms": float(hash_ms), "integrity_hash_ms": float(hash_ms),
            "file_fsync_ms": float(file_fsync_ms), "directory_fsync_ms": float(directory_fsync_ms),
            "atomic_rename_ms": float(atomic_rename_ms), "parent_fsync_ms": float(parent_fsync_ms),
            "durability_ms": float(durability_ms), "persist_ms": float(persist_ms),
            "helper_total_ms": (time.perf_counter() - persist_t0) * 1e3,
        }
        return {
            "store_dir": str(destination.resolve()), "image_id": str(image_id), "meta": meta, "timing_ms": timing_ms,
            "bytes": {"visual_kv": int(meta["bytes_visual_kv"]), "probe_sidecar": int(meta["bytes_probe_sidecar"]),
                      "separator_sidecar": 0, "system_prefix": int(file_sizes.get("sys_kv.pt", 0)),
                      "meta": int(file_sizes.get("meta.json", 0)), "other": total - known, "total": total},
            "file_sizes": file_sizes,
            "hashes": {"files_sha256": small_hashes, "prefix_kv_sample_sha256": sample_hash,
                       "identity_permutation_sha256": identity_sha},
            "durability": {"same_filesystem_staging": True, "atomic_no_clobber": True, "files_fsynced": int(synced_files),
                           "directories_fsynced_before_rename": int(synced_dirs), "parent_fsynced_after_rename": True},
        }
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)

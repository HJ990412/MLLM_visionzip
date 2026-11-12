"""PackedPrefixStore for Qwen2.5-VL PrefixStore-KV25 (``qwen25_bf16_prefix25_packed_v1``).

One image == one ``kv_packed.bin``::

    [structural K/V][L0 K rows 0..k_0)[L0 V rows 0..k_0)][L1 K][L1 V] ... [L27 V]

* The structural block is byte-identical to ``structural_kv.bin`` of the
  unpacked ``prefix25`` store (``(2, L, S, H, D)``: kind, layer, structural row).
* Every layer block is token-major ``(k_l, H, D)`` native BF16 in the global
  image-only stable importance order -- the same bytes the ``prefix25`` writer
  puts in ``layer_xxx/{k,v}.bin`` -- cut at the layer's own row count ``k_l``.
  No padding.
* ``k_l`` always comes from a frozen ``mmimpress.layer_budget`` document whose
  ``num_layers`` / ``total_ratio`` / ``r_min`` / ``r_max`` are passed
  explicitly (the module defaults are LLaVA's 32 layers / floor 0.10 and are
  never relied on here).  The uniform vector reproduces ``k_l = ceil(N/4)``.

Reading: ``QwenPackedStore.read_all`` issues exactly ONE ``os.preadv`` of the
whole file into a page-locked host buffer allocated at activation; ``os.pread``
is not used on the hit path.  Activation (outside every request timer) reads
metadata, validates it fail-closed and verifies the payload SHA256.

Contract: docs/qwen_layer_budget_pyr_contract.md.  This module only adds a
format; ``mmimpress/qwen25/store.py`` is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mmimpress.cvpr25 import permutation_sha256
from mmimpress.layer_budget import (POLICY, budget_sha256, layer_keep_counts,
                                    total_row_budget, validate_ratios)
from .store import (_REQUIRED_EXTRA, _ROW_ITEMSIZE, IOCounter, _bf16_bytes,
                    _canonical_json, _deep_size, _jsonable, _normalise_layers,
                    _normalise_scores, _position_ids, _pread_exact,
                    inverse_permutation, stable_visual_order)

STORE_FORMAT = "qwen25_bf16_prefix25_packed_v1"
STORAGE_POLICY = "prefix25_packed"
PACKED_NAME = "kv_packed.bin"
META_NAME = "meta.json"
TOTAL_RATIO = 0.25
_BUDGET_KEYS = ("alloc_id", "layer_keep_ratios", "layer_budget_sha256", "r_min",
                "r_max", "num_layers", "total_ratio", "policy")


# ------------------------------------------------------------------ budget
def budget_fields(doc: dict, num_layers: int) -> tuple[list[float], float, float]:
    """Explicit, fail-closed view of a frozen budget document for ``num_layers``."""
    if not isinstance(doc, dict):
        raise ValueError("packed stores require a frozen layer budget document")
    missing = [key for key in _BUDGET_KEYS if key not in doc]
    if missing:
        raise ValueError(f"layer budget document lacks {missing}")
    if doc["policy"] != POLICY:
        raise ValueError("unsupported layer budget policy")
    if isinstance(doc["num_layers"], bool) or int(doc["num_layers"]) != int(num_layers):
        raise ValueError("layer budget num_layers differs from the decoder")
    if float(doc["total_ratio"]) != TOTAL_RATIO:
        raise ValueError("packed stores serve only the 0.25 total ratio")
    r_min, r_max = float(doc["r_min"]), float(doc["r_max"])
    ratios = validate_ratios(doc["layer_keep_ratios"], num_layers=int(num_layers),
                             total_ratio=TOTAL_RATIO, r_min=r_min, r_max=r_max)
    if doc["layer_budget_sha256"] != budget_sha256(ratios):
        raise ValueError("layer budget SHA256 does not match its ratios")
    return ratios, r_min, r_max


def budget_binding_sha256(alloc_id: str, ratios, r_min: float, r_max: float,
                          num_layers: int, total_ratio: float) -> str:
    """Digest binding the vector to its clip range and layer count."""
    return hashlib.sha256(_canonical_json({
        "alloc_id": str(alloc_id), "layer_keep_ratios": [float(x) for x in ratios],
        "r_min": float(r_min), "r_max": float(r_max),
        "num_layers": int(num_layers), "total_ratio": float(total_ratio),
    })).hexdigest()


def packed_keep_counts(ratios, n_content: int, num_layers: int, r_min: float,
                       r_max: float) -> list[int]:
    """Per-layer row counts; every argument explicit, no zero-row layer."""
    counts = layer_keep_counts(ratios, int(n_content), num_layers=int(num_layers),
                               total_ratio=TOTAL_RATIO, r_min=float(r_min),
                               r_max=float(r_max))
    if sum(counts) != total_row_budget(int(n_content), num_layers=int(num_layers),
                                       total_ratio=TOTAL_RATIO):
        raise ValueError("layer keep counts do not sum to the row budget")
    if sum(counts) != int(num_layers) * ((int(n_content) + 3) // 4):
        raise ValueError("row budget differs from the uniform store")
    if min(counts) < 1:
        raise ValueError("a layer would keep zero content rows")
    return counts


# ------------------------------------------------------------------ layout
def packed_layout(counts, row_bytes: int, num_layers: int, structural_count: int):
    """Byte table: structural block first, then K/V per layer."""
    counts = [int(c) for c in counts]
    if len(counts) != int(num_layers) or any(c < 0 for c in counts):
        raise ValueError("invalid per-layer row counts")
    structural_bytes = 2 * int(num_layers) * int(structural_count) * int(row_bytes)
    off = structural_bytes
    table = []
    for li, k in enumerate(counts):
        k_off = off
        off += k * row_bytes
        v_off = off
        off += k * row_bytes
        table.append({"layer": li, "k_offset": k_off, "v_offset": v_off, "rows": k})
    return table, structural_bytes, off


# ------------------------------------------------------------------ writer
def write_qwen_packed_store(out_dir: str | Path, layers: Any, visual_start: int,
                            visual_count: int, prefix_ids: Any, scores: Any,
                            extra: dict, *, layer_budget: dict,
                            timing_out: dict | None = None) -> dict:
    """Write one packed image prefix directly from the Turn-1 capture.

    Row content and order are produced by the same operations as the
    ``prefix25`` writer in ``store.write_qwen_store``; only the container and
    the per-layer cut differ.  The destination must not exist.
    """
    started = time.perf_counter()
    if scores is None:
        raise ValueError("packed stores require image-only scores")
    visual_start, visual_count = int(visual_start), int(visual_count)
    prefix_ids = _jsonable(prefix_ids)
    if len(prefix_ids) == 1 and isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    prefix_ids = [int(x) for x in prefix_ids]
    prefix_len = len(prefix_ids)
    if visual_start < 0 or visual_count < 1 or visual_start + visual_count > prefix_len:
        raise ValueError("invalid expanded image-token interval in prefix")
    extra = _jsonable(extra or {})
    missing = [key for key in _REQUIRED_EXTRA if key not in extra]
    if missing:
        raise ValueError(f"missing Qwen store metadata: {missing}")
    for key in ("image_sha256", "checkpoint_revision", "processor_revision",
                "code_revision", "environment_revision", "position_policy",
                "key_rope_state"):
        if not isinstance(extra[key], str) or not extra[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if extra["key_rope_state"] != "post_mrope":
        raise ValueError("this store requires post-MRoPE cached keys")
    if len(extra["image_sha256"]) != 64 or any(
            ch not in "0123456789abcdefABCDEF" for ch in extra["image_sha256"]):
        raise ValueError("image_sha256 must be a 64-digit content digest")
    if not isinstance(extra["processor_settings"], dict) \
            or not isinstance(extra["geometry"], dict):
        raise ValueError("processor_settings and geometry must be objects")
    if extra.get("score_source") is None or extra.get("score_layer") is None:
        raise ValueError("packed store requires score_source and score_layer")
    position_ids = _position_ids(extra["logical_position_ids"], prefix_len)
    layers = _normalise_layers(layers, prefix_len)
    num_layers = len(layers)
    heads, head_dim = int(layers[0][0].shape[1]), int(layers[0][0].shape[3])
    score_values = _normalise_scores(scores, visual_count)
    order = stable_visual_order(score_values, visual_count)
    inverse = inverse_permutation(order)
    permutation_end = time.perf_counter()

    ratios, r_min, r_max = budget_fields(layer_budget, num_layers)
    counts = packed_keep_counts(ratios, visual_count, num_layers, r_min, r_max)
    structural_indices = [i for i in range(prefix_len)
                          if i < visual_start or i >= visual_start + visual_count]
    structural_count = len(structural_indices)
    row_bytes = heads * head_dim * _ROW_ITEMSIZE
    table, structural_bytes, total_bytes = packed_layout(
        counts, row_bytes, num_layers, structural_count)
    identity = {key: extra[key] for key in (
        "image_sha256", "checkpoint_revision", "processor_settings",
        "image_grid_thw", "geometry", "position_policy")}
    identity.update({"prefix_input_ids": prefix_ids, "kv_dtype": "bfloat16",
                     "processor_revision": extra["processor_revision"],
                     "code_revision": extra["code_revision"],
                     "environment_revision": extra["environment_revision"],
                     "key_rope_state": extra["key_rope_state"],
                     "logical_position_ids_sha256": hashlib.sha256(
                         _canonical_json(position_ids)).hexdigest()})
    identity_hash = hashlib.sha256(_canonical_json(identity)).hexdigest()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    materialize = repack = serialize = 0.0
    blocks = []
    # Structural rows: identical operations to store.write_qwen_store.
    t0 = time.perf_counter()
    struct_per_kind = []
    for kind_idx in range(2):
        struct_per_layer = []
        for pair in layers:
            tensor = pair[kind_idx]
            indices = torch.tensor(structural_indices, dtype=torch.long,
                                   device=tensor.device)
            rows = tensor[0].index_select(1, indices).permute(1, 0, 2)
            struct_per_layer.append(rows.to("cpu").contiguous())
        struct_per_kind.append(torch.stack(struct_per_layer))
    structural = torch.stack(struct_per_kind).contiguous()
    materialize += time.perf_counter() - t0
    blocks.append(_bf16_bytes(structural))
    if len(blocks[0]) != structural_bytes:
        raise AssertionError("structural block size mismatch")

    order_cpu = torch.tensor(order, dtype=torch.long)
    for li, pair in enumerate(layers):
        for tensor in pair:
            t0 = time.perf_counter()
            rows = tensor[0, :, visual_start:visual_start + visual_count, :]
            rows = rows.permute(1, 0, 2).to("cpu").contiguous()
            materialize += time.perf_counter() - t0
            t0 = time.perf_counter()
            rows = rows.index_select(0, order_cpu[:counts[li]]).contiguous()
            repack += time.perf_counter() - t0
            t0 = time.perf_counter()
            blocks.append(_bf16_bytes(rows))
            serialize += time.perf_counter() - t0
    t0 = time.perf_counter()
    blob = b"".join(blocks)
    serialize += time.perf_counter() - t0
    if len(blob) != total_bytes:
        raise AssertionError("packed payload size mismatch")
    t0 = time.perf_counter()
    packed_sha = hashlib.sha256(blob).hexdigest()
    hash_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    with (out_dir / PACKED_NAME).open("xb") as handle:
        handle.write(blob)
        handle.flush()
        write_end = time.perf_counter()
        os.fsync(handle.fileno())
    write_seconds = write_end - t0
    fsync_seconds = time.perf_counter() - write_end

    k_max = max(counts)
    meta = {
        "format": STORE_FORMAT, "store_format": STORE_FORMAT,
        "storage_policy": STORAGE_POLICY, "model_family": "Qwen2.5-VL",
        "layout": "packed_token_major_repacked_bf16",
        "layout_description": "kv_packed.bin = [structural (2,L,S,H,D)][L0 K rows 0..k_0)[L0 V]...[L(L-1) V]; "
                              "layer blocks token-major (k_l,H,D) BF16 in the global importance order; no padding",
        "dtype": "bfloat16", "byte_order": sys.byteorder,
        "num_layers": num_layers, "num_kv_heads": heads, "head_dim": head_dim,
        "native_kv_shape": [1, heads, prefix_len, head_dim],
        "visual_start": visual_start, "visual_count": visual_count,
        "prefix_len": prefix_len, "prefix_input_ids": prefix_ids,
        "prefix_sha256": hashlib.sha256(_canonical_json(prefix_ids)).hexdigest(),
        "structural_indices": structural_indices,
        "structural_count": structural_count, "row_bytes": row_bytes,
        "full_importance_permutation": order,
        "stored_to_original": order[:k_max],
        "stored_row_to_original": order[:k_max],
        "original_to_stored": inverse,
        "inverse_semantics": "original token to full importance rank, not stored row",
        "order_semantics": "layer l keeps full_importance_permutation[:layer_keep_counts[l]]",
        "permutation_sha256": permutation_sha256(order),
        "global_order_all_layers": True,
        "score_source": extra["score_source"], "score_layer": extra["score_layer"],
        "saliency_sha256": hashlib.sha256(
            b"".join(__import__("struct").pack("<f", x) for x in score_values)).hexdigest(),
        "image_grid_thw": extra["image_grid_thw"], "geometry": extra["geometry"],
        "logical_position_ids": position_ids,
        "key_rope_state": extra["key_rope_state"],
        "rope_deltas": extra.get("rope_deltas"),
        "identity": identity, "identity_sha256": identity_hash,
        "code_revision": extra["code_revision"],
        "environment_revision": extra["environment_revision"],
        "processor_revision": extra["processor_revision"],
        "packed_code_revision": extra.get("packed_code_revision"),
        "original_content_count": visual_count,
        "uniform_equivalent_content_count": (visual_count + 3) // 4,
        "retention_ratio": TOTAL_RATIO,
        "rounding_policy": "hamilton_largest_remainder_total_ceil_original_content",
        "budget_policy": POLICY,
        "layer_budget_id": str(layer_budget["alloc_id"]),
        "layer_keep_ratios": [float(x) for x in ratios],
        "layer_keep_counts": list(counts),
        "layer_budget_sha256": str(layer_budget["layer_budget_sha256"]),
        "layer_budget_r_min": r_min, "layer_budget_r_max": r_max,
        "layer_budget_total_ratio": TOTAL_RATIO,
        "layer_budget_binding_sha256": budget_binding_sha256(
            layer_budget["alloc_id"], ratios, r_min, r_max, num_layers, TOTAL_RATIO),
        "stored_content_count_total": int(sum(counts)),
        "packed_file": PACKED_NAME, "packed_layout": table,
        "packed_structural_offset": 0, "packed_structural_bytes": structural_bytes,
        "packed_total_bytes": total_bytes, "packed_sha256": packed_sha,
        "bytes_visual_kv": total_bytes - structural_bytes,
        "bytes_structural_kv": structural_bytes, "bytes_metadata_file": 0,
        "extra": {key: value for key, value in extra.items()
                  if key not in _REQUIRED_EXTRA},
    }
    for _ in range(10):
        encoded = (json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False,
                              allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) == meta["bytes_metadata_file"]:
            break
        meta["bytes_metadata_file"] = len(encoded)
    else:
        raise RuntimeError("metadata byte count did not converge")
    t0 = time.perf_counter()
    with (out_dir / META_NAME).open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        meta_write_end = time.perf_counter()
        os.fsync(handle.fileno())
    write_seconds += meta_write_end - t0
    fsync_seconds += time.perf_counter() - meta_write_end
    t0 = time.perf_counter()
    dir_fd = os.open(out_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    fsync_seconds += time.perf_counter() - t0
    if timing_out is not None:
        timing_out.update({
            "prefix_slice_serialization_ms": serialize * 1e3,
            "permutation_ms": (permutation_end - started) * 1e3,
            "kv_materialize_ms": materialize * 1e3,
            "kv_repack_ms": repack * 1e3,
            "packed_sha256_ms": hash_seconds * 1e3,
            "ssd_write_ms": write_seconds * 1e3,
            "fsync_ms": fsync_seconds * 1e3,
            "total_persistence_ms": (time.perf_counter() - started) * 1e3,
        })
    return meta


# --------------------------------------------------------------- validator
def validate_packed_meta(m: dict, *, expected_identity: dict | None = None,
                         expected_budget: dict | None = None,
                         metadata_file_bytes: int | None = None,
                         payload_file_bytes: int | None = None) -> None:
    """Fail-closed check of packed metadata: geometry, vector, counts, bytes."""
    if not isinstance(m, dict):
        raise ValueError("packed metadata must be an object")
    if (m.get("format") != STORE_FORMAT or m.get("store_format") != STORE_FORMAT
            or m.get("storage_policy") != STORAGE_POLICY
            or m.get("packed_file") != PACKED_NAME
            or m.get("layout") != "packed_token_major_repacked_bf16"
            or m.get("dtype") != "bfloat16" or m.get("byte_order") != sys.byteorder
            or m.get("key_rope_state") != "post_mrope"
            or m.get("budget_policy") != POLICY
            or m.get("global_order_all_layers") is not True):
        raise ValueError("unsupported or mislabeled Qwen packed store")
    for key in ("num_layers", "num_kv_heads", "head_dim", "visual_start",
                "visual_count", "prefix_len", "structural_count", "row_bytes",
                "packed_total_bytes", "packed_structural_bytes"):
        if isinstance(m.get(key), bool) or not isinstance(m.get(key), int):
            raise ValueError(f"packed metadata field {key} must be an integer")
    L, heads, hd = m["num_layers"], m["num_kv_heads"], m["head_dim"]
    n, p, start = m["visual_count"], m["prefix_len"], m["visual_start"]
    if min(L, heads, hd, n, p) < 1 or start < 0 or start + n > p:
        raise ValueError("invalid store dimensions")
    if m["row_bytes"] != heads * hd * _ROW_ITEMSIZE \
            or m.get("native_kv_shape") != [1, heads, p, hd]:
        raise ValueError("row byte width or native KV shape mismatch")
    # ---- vector, clip range, layer count (never the module defaults)
    ratios_raw = m.get("layer_keep_ratios")
    if not isinstance(ratios_raw, list) or len(ratios_raw) != L:
        raise ValueError("layer ratio vector length differs from num_layers")
    r_min, r_max = m.get("layer_budget_r_min"), m.get("layer_budget_r_max")
    if not isinstance(r_min, float) or not isinstance(r_max, float) \
            or m.get("layer_budget_total_ratio") != TOTAL_RATIO \
            or m.get("retention_ratio") != TOTAL_RATIO:
        raise ValueError("packed store lacks an explicit clip range / total ratio")
    ratios = validate_ratios(ratios_raw, num_layers=L, total_ratio=TOTAL_RATIO,
                             r_min=r_min, r_max=r_max)
    if m.get("layer_budget_sha256") != budget_sha256(ratios):
        raise ValueError("layer budget SHA256 does not match the stored ratios")
    if m.get("layer_budget_binding_sha256") != budget_binding_sha256(
            m.get("layer_budget_id"), ratios, r_min, r_max, L, TOTAL_RATIO):
        raise ValueError("layer budget binding digest mismatch")
    if expected_budget is not None:
        e_ratios, e_min, e_max = budget_fields(expected_budget, L)
        if (m["layer_budget_sha256"] != expected_budget["layer_budget_sha256"]
                or ratios != e_ratios or r_min != e_min or r_max != e_max
                or m.get("layer_budget_id") != str(expected_budget["alloc_id"])):
            raise ValueError("packed store was built with a different layer budget")
    counts = packed_keep_counts(ratios, n, L, r_min, r_max)
    stored_counts = m.get("layer_keep_counts")
    if not isinstance(stored_counts, list) or len(stored_counts) != L \
            or any(isinstance(c, bool) or not isinstance(c, int) for c in stored_counts) \
            or stored_counts != counts:
        raise ValueError("layer keep counts do not match the ratio vector and N")
    if m.get("original_content_count") != n \
            or m.get("uniform_equivalent_content_count") != (n + 3) // 4 \
            or m.get("stored_content_count_total") != sum(counts):
        raise ValueError("content count bookkeeping mismatch")
    # ---- order
    order = m.get("full_importance_permutation")
    if not isinstance(order, list) or len(order) != n:
        raise ValueError("invalid importance permutation length")
    if inverse_permutation(order) != m.get("original_to_stored"):
        raise ValueError("permutation inverse mismatch")
    if permutation_sha256(order) != m.get("permutation_sha256"):
        raise ValueError("permutation digest mismatch")
    k_max = max(counts)
    if m.get("stored_to_original") != order[:k_max] \
            or m.get("stored_row_to_original") != order[:k_max]:
        raise ValueError("stored row mapping mismatch")
    if m.get("saliency_sha256") is None or m.get("score_source") is None \
            or m.get("score_layer") is None:
        raise ValueError("packed store lacks saliency provenance")
    # ---- prefix / structural / positions / identity
    if not isinstance(m.get("prefix_input_ids"), list) or len(m["prefix_input_ids"]) != p:
        raise ValueError("invalid prefix IDs")
    if hashlib.sha256(_canonical_json(m["prefix_input_ids"])).hexdigest() != m.get("prefix_sha256"):
        raise ValueError("prefix token digest mismatch")
    expected_struct = [i for i in range(p) if i < start or i >= start + n]
    if m.get("structural_indices") != expected_struct \
            or m["structural_count"] != len(expected_struct):
        raise ValueError("structural prefix index mismatch")
    if _position_ids(m["logical_position_ids"], p) != m["logical_position_ids"]:
        raise ValueError("logical MRoPE position shape mismatch")
    identity = m.get("identity")
    if not isinstance(identity, dict) \
            or identity.get("prefix_input_ids") != m["prefix_input_ids"] \
            or identity.get("kv_dtype") != "bfloat16" \
            or identity.get("image_grid_thw") != m.get("image_grid_thw") \
            or identity.get("geometry") != m.get("geometry") \
            or identity.get("key_rope_state") != m["key_rope_state"] \
            or identity.get("code_revision") != m.get("code_revision") \
            or identity.get("environment_revision") != m.get("environment_revision") \
            or identity.get("logical_position_ids_sha256") != hashlib.sha256(
                _canonical_json(m["logical_position_ids"])).hexdigest():
        raise ValueError("cache identity contradicts store metadata")
    if hashlib.sha256(_canonical_json(identity)).hexdigest() != m.get("identity_sha256"):
        raise ValueError("cache identity digest mismatch")
    if expected_identity is not None:
        for key, expected in _jsonable(expected_identity).items():
            if identity.get(key) != expected:
                raise ValueError(f"cache identity mismatch: {key}")
    # ---- byte table
    table, structural_bytes, total = packed_layout(counts, m["row_bytes"], L,
                                                   m["structural_count"])
    if (m.get("packed_layout") != table
            or m.get("packed_structural_offset") != 0
            or m["packed_structural_bytes"] != structural_bytes
            or m["packed_total_bytes"] != total
            or m.get("bytes_structural_kv") != structural_bytes
            or m.get("bytes_visual_kv") != total - structural_bytes
            or total != structural_bytes + sum(2 * k * m["row_bytes"] for k in counts)):
        raise ValueError("packed byte table disagrees with the row counts")
    previous = structural_bytes
    for row in table:
        if not (row["k_offset"] == previous and row["v_offset"] == row["k_offset"]
                + row["rows"] * m["row_bytes"]):
            raise ValueError("packed offsets are not monotone and gapless")
        previous = row["v_offset"] + row["rows"] * m["row_bytes"]
    if not isinstance(m.get("packed_sha256"), str) or len(m["packed_sha256"]) != 64:
        raise ValueError("packed payload digest missing")
    if metadata_file_bytes is not None and m.get("bytes_metadata_file") != metadata_file_bytes:
        raise ValueError("metadata byte count mismatch")
    if payload_file_bytes is not None and payload_file_bytes != total:
        raise ValueError("packed file length disagrees with metadata")


# ------------------------------------------------------------------ reader
@dataclass
class PackedPlan:
    """Row gather plan: flat row table -> per-layer [structural + first k_l rows]
    in original-position order.  Pure function of the metadata."""
    row_index: torch.Tensor              # (sum_l 2*(S+k_l),) rows of the flat table
    spans: list[tuple[int, int, int]]    # per layer: K start, V start, end
    lengths: list[int]                   # per layer: S + k_l
    counts: list[int]
    planning_ms: float


class QwenPackedStore:
    """Activated packed image store; metadata, one FD and a host buffer."""

    def __init__(self, path: str | Path, expected_identity: dict | None = None,
                 expected_budget: dict | None = None, verify: bool = True,
                 pin: bool = True):
        self.path = Path(path)
        self.activation_io = IOCounter()
        started = time.perf_counter()
        self._fd = None
        self.buf = None
        self._np = None
        try:
            fd = os.open(self.path / META_NAME, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                raw = _pread_exact(fd, size, 0, self.activation_io, "metadata", META_NAME)
            finally:
                os.close(fd)
            self.metadata_file_bytes = len(raw)
            self.metadata_file_sha256 = hashlib.sha256(raw).hexdigest()
            self.meta = json.loads(raw)
            self.metadata_resident_bytes = _deep_size(self.meta)
            payload = self.path / PACKED_NAME
            if payload.is_symlink() or not payload.is_file():
                raise ValueError("packed payload is missing")
            validate_packed_meta(self.meta, expected_identity=expected_identity,
                                 expected_budget=expected_budget,
                                 metadata_file_bytes=self.metadata_file_bytes,
                                 payload_file_bytes=payload.stat().st_size)
            self.total = int(self.meta["packed_total_bytes"])
            self._fd = os.open(payload, os.O_RDONLY)
            if verify:
                self._verify_payload_hash()
            self.buf = torch.empty(self.total, dtype=torch.uint8)
            self.pinned = False
            if pin and torch.cuda.is_available():
                self.buf = self.buf.pin_memory()
                self.pinned = True
            self._np = self.buf.numpy()
            self.read_calls = 0
            self.activation_ms = (time.perf_counter() - started) * 1e3
        except Exception:
            self.close()
            raise

    def _verify_payload_hash(self) -> None:
        digest = hashlib.sha256()
        cursor = 0
        while cursor < self.total:
            count = min(4 * 1024 * 1024, self.total - cursor)
            payload = _pread_exact(self._fd, count, cursor, self.activation_io,
                                   "activation_packed", PACKED_NAME)
            digest.update(payload)
            cursor += len(payload)
        if digest.hexdigest() != self.meta["packed_sha256"]:
            raise ValueError("packed payload hash mismatch")

    # -- request-time pieces -------------------------------------------------
    def plan(self) -> PackedPlan:
        started = time.perf_counter()
        m = self.meta
        L, S = int(m["num_layers"]), int(m["structural_count"])
        pre = int(m["visual_start"])           # structural rows before the image
        row_bytes = int(m["row_bytes"])
        counts = [int(c) for c in m["layer_keep_counts"]]
        k_max = max(counts)
        order = torch.tensor(m["full_importance_permutation"][:k_max], dtype=torch.long)
        stored_sorted = torch.argsort(order)   # stored rows by ascending original position
        ar = torch.arange(S, dtype=torch.long)
        parts, spans, lengths, cursor = [], [], [], 0
        for li, row in enumerate(m["packed_layout"]):
            k = counts[li]
            selected = stored_sorted[stored_sorted < k]
            for kind_idx, key in enumerate(("k_offset", "v_offset")):
                base = int(row[key]) // row_bytes
                sbase = (kind_idx * L + li) * S
                parts.extend((ar[:pre] + sbase, selected + base, ar[pre:] + sbase))
            spans.append((cursor, cursor + S + k, cursor + 2 * (S + k)))
            lengths.append(S + k)
            cursor += 2 * (S + k)
        row_index = torch.cat(parts)
        return PackedPlan(row_index=row_index, spans=spans, lengths=lengths,
                          counts=counts,
                          planning_ms=(time.perf_counter() - started) * 1e3)

    def read_all(self, io: IOCounter | None = None) -> int:
        """The single timed read: whole file -> host buffer, one ``preadv``."""
        if io is not None:
            io.span("packed", PACKED_NAME, 0, self.total)
        started = time.perf_counter()
        n = os.preadv(self._fd, [memoryview(self._np)], 0)
        elapsed = time.perf_counter() - started
        self.read_calls += 1
        if io is not None:
            io.call("packed", n, elapsed)
        if n != self.total:
            raise EOFError(f"short packed read: {n} != {self.total}")
        return n

    def rows(self) -> torch.Tensor:
        """Zero-copy ``(total_rows, H, D)`` BF16 view of the host buffer."""
        m = self.meta
        return self.buf.view(torch.bfloat16).view(-1, int(m["num_kv_heads"]),
                                                  int(m["head_dim"]))

    def assemble_cpu(self, plan: PackedPlan | None = None):
        """CPU assembly of the per-layer ``[1, H, S+k_l, D]`` pairs (tests, gates)."""
        plan = plan or self.plan()
        gathered = self.rows().index_select(0, plan.row_index)
        return [(gathered[a:b].permute(1, 0, 2).unsqueeze(0).contiguous(),
                 gathered[b:c].permute(1, 0, 2).unsqueeze(0).contiguous())
                for a, b, c in plan.spans]

    # -- diagnostics (never timed) ------------------------------------------
    def layer_selected_original(self, layer: int) -> list[int]:
        m = self.meta
        return sorted(int(x) for x in
                      m["full_importance_permutation"][:int(m["layer_keep_counts"][layer])])

    def layer_logical_indices(self, layer: int) -> list[int]:
        m = self.meta
        start = int(m["visual_start"])
        return sorted(list(m["structural_indices"])
                      + [start + i for i in self.layer_selected_original(layer)])

    def drop_payload_cache(self) -> dict:
        """Request cold-page conditioning; caller times this outside TTFT."""
        if not callable(getattr(os, "posix_fadvise", None)) \
                or not hasattr(os, "POSIX_FADV_DONTNEED"):
            return {"attempted": 1, "failed": [{"path": PACKED_NAME,
                                                "error": "posix_fadvise unavailable"}]}
        try:
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            return {"attempted": 1, "failed": [{"path": PACKED_NAME, "error": str(exc)}]}
        return {"attempted": 1, "failed": []}

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._np = None
        self.buf = None

    def __enter__(self) -> "QwenPackedStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def open_qwen_packed_store(path: str | Path, expected_identity: dict | None = None,
                           expected_budget: dict | None = None, verify: bool = True,
                           pin: bool = True) -> QwenPackedStore:
    return QwenPackedStore(path, expected_identity=expected_identity,
                           expected_budget=expected_budget, verify=verify, pin=pin)

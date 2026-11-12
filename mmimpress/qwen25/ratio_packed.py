"""Ratio-generalised uniform PackedPrefixStore for Qwen2.5-VL
(``qwen25_bf16_prefix_ratio_packed_v1``).

``mmimpress/qwen25/packed_store.py`` (PyramidKV run) serves only a total ratio
of 0.25: its writer, validator and budget view hard-code ``28 * ceil(N/4)``.
This module adds the same container for any total ratio in ``(0, 1]`` taken
from the frozen budget document, for UNIFORM vectors only (every decoder layer
keeps the same ``k = ceil(ratio * N)`` content rows):

    kv_packed.bin = [structural K/V (2,L,S,H,D)][L0 K rows 0..k)[L0 V] ... [L27 V]

* Row content and row order are produced by exactly the operations of
  ``packed_store.write_qwen_packed_store``; at ratio 0.25 the payload is
  byte-identical to that writer's uniform store (gate SG2).
* ``k`` is computed twice: by ``mmimpress.layer_budget.layer_keep_counts``
  with every argument explicit (``num_layers``, ``total_ratio``, ``r_min``,
  ``r_max`` come from the document, never from the LLaVA module defaults) and
  by exact integer arithmetic on the decimal fraction of the ratio
  (``ceil(p * N / q)``).  A disagreement fails closed.
* Reading, planning, H2D, GPU assembly, the per-layer mask hooks and the
  suffix forward are NOT re-implemented: ``RatioPackedStore`` inherits the
  reader of ``QwenPackedStore`` and ``run_cache_ratio`` calls
  ``layerwise.run_cache_layerwise`` unchanged (one ``preadv`` per hit).

Nothing here modifies ``packed_store.py`` / ``layerwise.py`` / ``store.py`` /
``runner.py``.  Contract: docs/qwen_budget_sweep_contract.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import torch

from mmimpress.cvpr25 import permutation_sha256
from mmimpress.layer_budget import (POLICY, budget_sha256, layer_keep_counts,
                                    total_row_budget, validate_ratios)
from . import layerwise as LW
from .packed_store import (META_NAME, PACKED_NAME, QwenPackedStore,
                           budget_binding_sha256, packed_layout)
from .runner import POSITION_POLICY, _ms
from .store import (_REQUIRED_EXTRA, _ROW_ITEMSIZE, IOCounter, _bf16_bytes,
                    _canonical_json, _deep_size, _jsonable, _normalise_layers,
                    _normalise_scores, _position_ids, _pread_exact,
                    inverse_permutation, stable_visual_order)

STORE_FORMAT = "qwen25_bf16_prefix_ratio_packed_v1"
STORAGE_POLICY = "prefix_ratio_packed"
LAYOUT = "packed_token_major_repacked_bf16"
RATIO_CODE_FILES = ("packed_store.py", "layerwise.py", "ratio_packed.py")
_BUDGET_KEYS = ("alloc_id", "layer_keep_ratios", "layer_budget_sha256", "r_min",
                "r_max", "num_layers", "total_ratio", "policy")


def ratio_code_revision() -> str:
    """Digest of every file the ratio-packed write and hit paths execute."""
    here = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join(
        hashlib.sha256((here / name).read_bytes()).digest()
        for name in RATIO_CODE_FILES)).hexdigest()


# ------------------------------------------------------------------ budget
def ratio_fraction(total_ratio) -> Fraction:
    """Exact rational of the ratio's shortest decimal representation."""
    if isinstance(total_ratio, bool) or not isinstance(total_ratio, (int, float)):
        raise ValueError("total_ratio must be a real number")
    r = float(total_ratio)
    if not 0.0 < r <= 1.0:
        raise ValueError("total_ratio must be in (0, 1]")
    return Fraction(repr(r))


def uniform_keep_count(total_ratio, n_content: int) -> int:
    """``ceil(ratio * N)`` in exact integer arithmetic."""
    if isinstance(n_content, bool) or not isinstance(n_content, int) or n_content < 1:
        raise ValueError("n_content must be a positive integer")
    f = ratio_fraction(total_ratio)
    return -((-f.numerator * int(n_content)) // f.denominator)


def budget_fields(doc: dict, num_layers: int) -> tuple[list[float], float, float, float]:
    """Explicit, fail-closed view of a frozen UNIFORM budget document."""
    if not isinstance(doc, dict):
        raise ValueError("ratio-packed stores require a frozen layer budget document")
    missing = [key for key in _BUDGET_KEYS if key not in doc]
    if missing:
        raise ValueError(f"layer budget document lacks {missing}")
    if doc["policy"] != POLICY:
        raise ValueError("unsupported layer budget policy")
    if isinstance(doc["num_layers"], bool) or int(doc["num_layers"]) != int(num_layers):
        raise ValueError("layer budget num_layers differs from the decoder")
    total_ratio = float(doc["total_ratio"])
    ratio_fraction(total_ratio)
    r_min, r_max = float(doc["r_min"]), float(doc["r_max"])
    ratios = validate_ratios(doc["layer_keep_ratios"], num_layers=int(num_layers),
                             total_ratio=total_ratio, r_min=r_min, r_max=r_max)
    if any(r != total_ratio for r in ratios):
        raise ValueError("ratio-packed stores serve uniform vectors only")
    if doc["layer_budget_sha256"] != budget_sha256(ratios):
        raise ValueError("layer budget SHA256 does not match its ratios")
    return ratios, r_min, r_max, total_ratio


def ratio_keep_counts(ratios, n_content: int, num_layers: int, r_min: float,
                      r_max: float, total_ratio: float) -> list[int]:
    """Per-layer row counts: the layer-budget module with explicit arguments,
    cross-checked against the exact integer ``ceil(ratio * N)``."""
    n, L = int(n_content), int(num_layers)
    counts = layer_keep_counts(ratios, n, num_layers=L, total_ratio=float(total_ratio),
                               r_min=float(r_min), r_max=float(r_max))
    if sum(counts) != total_row_budget(n, num_layers=L, total_ratio=float(total_ratio)):
        raise ValueError("layer keep counts do not sum to the row budget")
    k = uniform_keep_count(total_ratio, n)
    if counts != [k] * L:
        raise ValueError("float row budget differs from the exact integer ceil(ratio * N)")
    if not 1 <= k <= n:
        raise ValueError("a layer would keep zero rows or more rows than exist")
    return counts


# ------------------------------------------------------------------ writer
def write_ratio_packed_store(out_dir: str | Path, layers: Any, visual_start: int,
                             visual_count: int, prefix_ids: Any, scores: Any,
                             extra: dict, *, layer_budget: dict,
                             timing_out: dict | None = None) -> dict:
    """Write one uniform packed image prefix directly from the Turn-1 capture.

    Same row operations, in the same order, as
    ``packed_store.write_qwen_packed_store``; only the row count per layer
    (``ceil(total_ratio * N)`` from the document) and the metadata differ.
    The destination must not exist.
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

    ratios, r_min, r_max, total_ratio = budget_fields(layer_budget, num_layers)
    counts = ratio_keep_counts(ratios, visual_count, num_layers, r_min, r_max, total_ratio)
    fraction = ratio_fraction(total_ratio)
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
    # Structural rows: identical operations to packed_store / store.write_qwen_store.
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

    k = counts[0]
    meta = {
        "format": STORE_FORMAT, "store_format": STORE_FORMAT,
        "storage_policy": STORAGE_POLICY, "model_family": "Qwen2.5-VL",
        "layout": LAYOUT,
        "layout_description": "kv_packed.bin = [structural (2,L,S,H,D)][L0 K rows 0..k)[L0 V]...[L(L-1) V]; "
                              "layer blocks token-major (k,H,D) BF16 in the global importance order; "
                              "k = ceil(total_ratio * N) for every layer; no padding",
        "dtype": "bfloat16", "byte_order": sys.byteorder,
        "num_layers": num_layers, "num_kv_heads": heads, "head_dim": head_dim,
        "native_kv_shape": [1, heads, prefix_len, head_dim],
        "visual_start": visual_start, "visual_count": visual_count,
        "prefix_len": prefix_len, "prefix_input_ids": prefix_ids,
        "prefix_sha256": hashlib.sha256(_canonical_json(prefix_ids)).hexdigest(),
        "structural_indices": structural_indices,
        "structural_count": structural_count, "row_bytes": row_bytes,
        "full_importance_permutation": order,
        "stored_to_original": order[:k],
        "stored_row_to_original": order[:k],
        "original_to_stored": inverse,
        "inverse_semantics": "original token to full importance rank, not stored row",
        "order_semantics": "every layer keeps full_importance_permutation[:uniform_content_count]",
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
        "uniform_content_count": k,
        "retention_ratio": total_ratio,
        "retention_fraction": [fraction.numerator, fraction.denominator],
        "rounding_policy": "uniform ceil(total_ratio * N); float layer_budget result must equal the exact integer fraction",
        "budget_policy": POLICY,
        "layer_budget_id": str(layer_budget["alloc_id"]),
        "layer_keep_ratios": [float(x) for x in ratios],
        "layer_keep_counts": list(counts),
        "layer_budget_sha256": str(layer_budget["layer_budget_sha256"]),
        "layer_budget_r_min": r_min, "layer_budget_r_max": r_max,
        "layer_budget_total_ratio": total_ratio,
        "layer_budget_binding_sha256": budget_binding_sha256(
            layer_budget["alloc_id"], ratios, r_min, r_max, num_layers, total_ratio),
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
def validate_ratio_packed_meta(m: dict, *, expected_identity: dict | None = None,
                               expected_budget: dict | None = None,
                               metadata_file_bytes: int | None = None,
                               payload_file_bytes: int | None = None) -> None:
    """Fail-closed check of ratio-packed metadata: geometry, ratio, counts, bytes."""
    if not isinstance(m, dict):
        raise ValueError("packed metadata must be an object")
    if (m.get("format") != STORE_FORMAT or m.get("store_format") != STORE_FORMAT
            or m.get("storage_policy") != STORAGE_POLICY
            or m.get("packed_file") != PACKED_NAME
            or m.get("layout") != LAYOUT
            or m.get("dtype") != "bfloat16" or m.get("byte_order") != sys.byteorder
            or m.get("key_rope_state") != "post_mrope"
            or m.get("budget_policy") != POLICY
            or m.get("global_order_all_layers") is not True):
        raise ValueError("unsupported or mislabeled Qwen ratio-packed store")
    for key in ("num_layers", "num_kv_heads", "head_dim", "visual_start",
                "visual_count", "prefix_len", "structural_count", "row_bytes",
                "packed_total_bytes", "packed_structural_bytes", "uniform_content_count"):
        if isinstance(m.get(key), bool) or not isinstance(m.get(key), int):
            raise ValueError(f"packed metadata field {key} must be an integer")
    L, heads, hd = m["num_layers"], m["num_kv_heads"], m["head_dim"]
    n, p, start = m["visual_count"], m["prefix_len"], m["visual_start"]
    if min(L, heads, hd, n, p) < 1 or start < 0 or start + n > p:
        raise ValueError("invalid store dimensions")
    if m["row_bytes"] != heads * hd * _ROW_ITEMSIZE \
            or m.get("native_kv_shape") != [1, heads, p, hd]:
        raise ValueError("row byte width or native KV shape mismatch")
    # ---- ratio, vector, clip range, layer count (never the module defaults)
    ratios_raw = m.get("layer_keep_ratios")
    if not isinstance(ratios_raw, list) or len(ratios_raw) != L:
        raise ValueError("layer ratio vector length differs from num_layers")
    r_min, r_max = m.get("layer_budget_r_min"), m.get("layer_budget_r_max")
    total_ratio = m.get("layer_budget_total_ratio")
    if not isinstance(r_min, float) or not isinstance(r_max, float) \
            or not isinstance(total_ratio, float) \
            or m.get("retention_ratio") != total_ratio:
        raise ValueError("packed store lacks an explicit clip range / total ratio")
    fraction = ratio_fraction(total_ratio)
    if m.get("retention_fraction") != [fraction.numerator, fraction.denominator]:
        raise ValueError("retention fraction does not match the total ratio")
    ratios = validate_ratios(ratios_raw, num_layers=L, total_ratio=total_ratio,
                             r_min=r_min, r_max=r_max)
    if any(r != total_ratio for r in ratios):
        raise ValueError("ratio-packed store is not uniform at its total ratio")
    if m.get("layer_budget_sha256") != budget_sha256(ratios):
        raise ValueError("layer budget SHA256 does not match the stored ratios")
    if m.get("layer_budget_binding_sha256") != budget_binding_sha256(
            m.get("layer_budget_id"), ratios, r_min, r_max, L, total_ratio):
        raise ValueError("layer budget binding digest mismatch")
    if expected_budget is not None:
        e_ratios, e_min, e_max, e_total = budget_fields(expected_budget, L)
        if (m["layer_budget_sha256"] != expected_budget["layer_budget_sha256"]
                or ratios != e_ratios or r_min != e_min or r_max != e_max
                or total_ratio != e_total
                or m.get("layer_budget_id") != str(expected_budget["alloc_id"])):
            raise ValueError("packed store was built with a different budget / ratio")
    counts = ratio_keep_counts(ratios, n, L, r_min, r_max, total_ratio)
    k = counts[0]
    stored_counts = m.get("layer_keep_counts")
    if not isinstance(stored_counts, list) or len(stored_counts) != L \
            or any(isinstance(c, bool) or not isinstance(c, int) for c in stored_counts) \
            or stored_counts != counts:
        raise ValueError("layer keep counts do not match the ratio and N")
    if m.get("original_content_count") != n \
            or m["uniform_content_count"] != k \
            or m.get("stored_content_count_total") != L * k:
        raise ValueError("content count bookkeeping mismatch")
    # ---- order
    order = m.get("full_importance_permutation")
    if not isinstance(order, list) or len(order) != n:
        raise ValueError("invalid importance permutation length")
    if inverse_permutation(order) != m.get("original_to_stored"):
        raise ValueError("permutation inverse mismatch")
    if permutation_sha256(order) != m.get("permutation_sha256"):
        raise ValueError("permutation digest mismatch")
    if m.get("stored_to_original") != order[:k] \
            or m.get("stored_row_to_original") != order[:k]:
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
            or total != structural_bytes + L * 2 * k * m["row_bytes"]):
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
class RatioPackedStore(QwenPackedStore):
    """Activated ratio-packed store.  Only activation (metadata validation)
    differs from ``QwenPackedStore``; ``plan`` / ``read_all`` / ``rows`` /
    ``assemble_cpu`` / ``drop_payload_cache`` / ``close`` are inherited."""

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
            validate_ratio_packed_meta(self.meta, expected_identity=expected_identity,
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


def open_ratio_packed_store(path: str | Path, expected_identity: dict | None = None,
                            expected_budget: dict | None = None, verify: bool = True,
                            pin: bool = True) -> RatioPackedStore:
    return RatioPackedStore(path, expected_identity=expected_identity,
                            expected_budget=expected_budget, verify=verify, pin=pin)


# ------------------------------------------------------------- persistence
def persist_ratio_packed(runner, capture, store_dir: str | Path, layer_budget: dict) -> dict:
    """Ratio analogue of ``layerwise.persist_packed`` (same receipt fields)."""
    t0 = time.perf_counter()
    timing = {}
    extra = {
        "image_sha256": capture.image_sha256,
        "processor_revision": runner.revision,
        "code_revision": runner._code_revision(),
        "environment_revision": runner._environment_revision(),
        "key_rope_state": "post_mrope",
        "checkpoint_revision": runner.revision,
        "processor_settings": runner.processor_settings(),
        "image_grid_thw": capture.image_grid_thw,
        "geometry": capture.geometry,
        "position_policy": POSITION_POLICY,
        "logical_position_ids": capture.logical_position_ids,
        "rope_deltas": capture.rope_deltas,
        "score_source": "last_fullatt_ViT_received_attention",
        "score_layer": runner.model.config.vision_config.depth - 1,
        "capture_score_ms": capture.score_ms,
        "packed_code_revision": ratio_code_revision(),
    }
    meta = write_ratio_packed_store(
        Path(store_dir), capture.layers, capture.visual_start,
        capture.visual_count, capture.prefix_ids, capture.scores, extra,
        layer_budget=layer_budget, timing_out=timing)
    writer_ms = _ms(t0)
    return {"store_dir": str(store_dir), "layout": "ratio_packed_repacked",
            "persistence_ms": writer_ms + capture.score_ms + capture.capture_clone_ms,
            "writer_persistence_ms": writer_ms, "timing_ms": timing,
            "capture_score_ms": capture.score_ms,
            "capture_clone_ms": capture.capture_clone_ms,
            "metadata": meta}


# -------------------------------------------------------------- activation
class RatioPackedSession(LW.PackedSession):
    """Registry of activated ratio-packed stores (per-image lifetime).
    ``condition`` / ``close`` are inherited from ``PackedSession``."""

    def _validate(self, store: RatioPackedStore, image_sha256: str, layer_budget: dict):
        runner, meta = self.runner, store.meta
        identity = meta["identity"]
        expected = {"checkpoint_revision": runner.revision,
                    "processor_settings": runner.processor_settings(),
                    "position_policy": POSITION_POLICY,
                    "image_sha256": image_sha256}
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"cache identity mismatch for {key}")
        if meta["dtype"] != "bfloat16":
            raise AssertionError("Qwen store must retain native BF16 bits")
        if meta.get("code_revision") != runner._code_revision():
            raise ValueError("packed store core code revision differs from the active runner")
        if meta.get("packed_code_revision") != ratio_code_revision():
            raise ValueError("packed store was written by different ratio-packed code")
        if meta.get("environment_revision") != runner._environment_revision():
            raise ValueError("cache environment revision differs from active runtime")
        ratios, r_min, r_max, total_ratio = budget_fields(layer_budget, int(meta["num_layers"]))
        if (meta["layer_budget_sha256"] != layer_budget["layer_budget_sha256"]
                or meta["layer_keep_ratios"] != ratios
                or meta["layer_budget_r_min"] != r_min
                or meta["layer_budget_r_max"] != r_max
                or meta["layer_budget_total_ratio"] != total_ratio
                or meta["layer_budget_id"] != str(layer_budget["alloc_id"])):
            raise ValueError("packed store budget / ratio differs from the request's")
        if int(meta["num_layers"]) != len(LW.decoder_layers(runner.model)):
            raise ValueError("packed store layer count differs from the decoder")

    def activate(self, store_dir: str | Path, image_sha256: str | None,
                 layer_budget: dict) -> RatioPackedStore:
        if image_sha256 is None:
            raise ValueError("cache activation requires image_sha256")
        key = str(Path(store_dir).resolve())
        if key not in self.stores:
            runner = self.runner
            expected_identity = {
                "image_sha256": image_sha256,
                "checkpoint_revision": runner.revision,
                "processor_settings": runner.processor_settings(),
                "position_policy": POSITION_POLICY,
                "environment_revision": runner._environment_revision(),
            }
            t0 = time.perf_counter()
            store = open_ratio_packed_store(Path(store_dir),
                                            expected_identity=expected_identity,
                                            expected_budget=layer_budget, verify=True)
            try:
                self._validate(store, image_sha256, layer_budget)
            except Exception:
                store.close()
                raise
            self.stores[key] = store
            self.activation_records[key] = {
                "activation_ms": _ms(t0),
                "activation_io": store.activation_io.summary(),
                "metadata_resident_bytes": store.metadata_resident_bytes,
                "pinned": store.pinned, "pinned_host_bytes": store.total,
                "metadata_file_sha256": store.metadata_file_sha256,
            }
        else:
            store = self.stores[key]
            self._validate(store, image_sha256, layer_budget)
        return store


# --------------------------------------------------------------------- hit
def run_cache_ratio(runner, session: RatioPackedSession, store_dir: str | Path,
                    question: str, history: Sequence[tuple[str, str]] = (), *,
                    layer_budget: dict, image_sha256: str | None = None,
                    return_logits: bool = False) -> dict:
    """Uniform ratio-packed cache hit.

    The whole timed path (prompt, plan, one ``preadv``, H2D, GPU assembly,
    mask-hook install, suffix prefill, decode) is ``run_cache_layerwise`` of
    the PyramidKV run, unchanged; this wrapper only supplies the ratio-aware
    session and relabels result fields after the request has finished.
    """
    if not isinstance(session, RatioPackedSession):
        raise TypeError("run_cache_ratio requires a RatioPackedSession")
    result = LW.run_cache_layerwise(runner, session, store_dir, question, history,
                                    layer_budget=layer_budget, image_sha256=image_sha256,
                                    return_logits=return_logits)
    meta = session.stores[str(Path(store_dir).resolve())].meta
    counts = result["layer_keep_counts"]
    k = int(meta["uniform_content_count"])
    if counts != [k] * int(meta["num_layers"]) or result["store_format"] != STORE_FORMAT:
        raise AssertionError("ratio-packed hit served a non-uniform or foreign store")
    quarter = result.pop("k_uniform_equivalent")
    result.update({
        "budget_unit": "visual_kv_uniform_ratio",
        "k_uniform": k,
        "k_quarter_reference": quarter,
        "retention_ratio": float(meta["retention_ratio"]),
        "retention_fraction": list(meta["retention_fraction"]),
        "selection_rule": "every layer attends original ids selected_importance_prefix_original[:k_uniform]",
        "hit_path": "mmimpress.qwen25.layerwise.run_cache_layerwise (unchanged) via RatioPackedSession",
    })
    return result

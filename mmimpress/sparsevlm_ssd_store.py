"""Read-only canonical KV store for the two explicit SparseVLM SSD policies.

Metadata activation is separate from request-time visual payload acquisition.
Each request has an event ledger over actual os.pread return lengths. AllHead
reads full K exactly once per layer and reuses it, then reads V chunks only.
Probe3 reads exactly the raw first three K heads and whole selected K/V chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from mmimpress.sparsevlm_ssd_core import canonical_chunk_plan, scoring_head_ids
from mmimpress.store import ChunkReader, merge_ranges


@dataclass
class LayerPayload:
    rows: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    stats: dict


class _EventReader(ChunkReader):
    """Reuse canonical addressing/coalescing with per-OS-call observations."""

    def __init__(self, directory, meta, events, probe_root=None):
        super().__init__(directory, meta, drop_cache=False)
        self.events = events
        self.category = None
        self.probe_root = Path(probe_root) if probe_root else self.dir

    def _fd(self, layer, name):
        if name != "probe_k" or self.probe_root == self.dir:
            return super()._fd(layer, name)
        key = (layer, name)
        if key not in self._fds:
            self._fds[key] = os.open(self.probe_root / f"layer_{layer:02d}" / "probe_k.bin", os.O_RDONLY)
        return self._fds[key]

    def _read(self, layer, name, ranges, counter, kind, units, max_gap=0):
        if max_gap != 0:
            raise ValueError("gap overread is forbidden")
        if self.category is None:
            raise RuntimeError("read requires one disjoint accounting category")
        fd = self._fd(layer, name)
        blobs = []
        for offset, length in merge_ranges(ranges, max_gap=0):
            started = time.perf_counter()
            blob = os.pread(fd, length, offset)
            elapsed = time.perf_counter() - started
            self.events.append({
                "layer": int(layer), "file": name + ".bin",
                "category": self.category, "offset": int(offset),
                "requested_bytes": int(length), "returned_bytes": len(blob),
                "preads": 1, "seconds": elapsed,
            })
            if len(blob) != length:
                raise IOError(f"short pread: layer={layer}, file={name}, actual={len(blob)}, expected={length}")
            blobs.append((offset, blob))
        return blobs


class CanonicalContext:
    """Metadata-ready CPU context; never retains visual/probe/selected KV.

    ``validate_provenance=False`` is strictly for synthetic CPU fixtures. The
    real serving path requires same-normal-T1 capture metadata and a raster
    identity layout. No AllHead method checks or opens a probe sidecar.
    """

    def __init__(self, path, head_policy, probe_root=None,
                 validate_provenance=True, expected_hashes=None):
        started = time.perf_counter()
        self.dir = Path(path)
        self.head_policy = head_policy
        self.probe_root = Path(probe_root) if probe_root else self.dir
        self.meta = json.loads((self.dir / "meta.json").read_text())
        m = self.meta
        self.head_ids = scoring_head_ids(head_policy, int(m["num_heads"]), int(m.get("num_key_value_heads", m["num_heads"])))
        self._validate_layout(validate_provenance)
        self.sys_kv = torch.load(self.dir / "sys_kv.pt", map_location="cpu", weights_only=True)
        self.v_hidden = torch.load(self.dir / "v_hidden.pt", map_location="cpu", weights_only=True)
        expected_sys = (int(m["num_layers"]), int(m["num_heads"]), int(m["v_token_start"]), int(m["head_dim"]))
        if set(self.sys_kv) != {"k", "v"}:
            raise ValueError("system metadata must contain exactly K and V")
        expected_dtype = {"float16": torch.float16, "float32": torch.float32}[m["dtype"]]
        if any(tuple(t.shape) != expected_sys or t.dtype != expected_dtype for t in self.sys_kv.values()):
            raise ValueError("system K/V shape or storage dtype mismatch")
        if not torch.is_tensor(self.v_hidden) or self.v_hidden.ndim != 2 or self.v_hidden.shape[0] != m["v_token_num"]:
            raise ValueError("v_hidden must contain the original expanded visual input rows")
        if not bool(torch.isfinite(self.v_hidden).all()):
            raise ValueError("v_hidden contains NaN/Inf")
        self.activation = {
            "seconds": time.perf_counter() - started,
            "file_bytes": sum((self.dir / name).stat().st_size for name in ("meta.json", "sys_kv.pt", "v_hidden.pt")),
            "host_tensor_bytes": sum(t.numel() * t.element_size() for t in self.sys_kv.values()) + self.v_hidden.numel() * self.v_hidden.element_size(),
            "gpu_tensor_bytes": 0,
            "visual_payload_resident_bytes": 0,
        }
        self.activation["provenance_validation"] = "ENFORCED" if validate_provenance else "SYNTHETIC_BYPASS"
        self.activation.update(hash_verification="NOT_REQUESTED", hash_verified_bytes=0,
                               hash_verification_seconds=0.0, verified_hashes={})
        if expected_hashes is not None:
            hash_started = time.perf_counter()
            required = {"meta.json", "sys_kv.pt", "v_hidden.pt", "sep_kv.bin"}
            for li in range(int(m["num_layers"])):
                required.update(f"layer_{li:02d}/{kind}.bin" for kind in ("k", "v"))
                if head_policy == "fixed_first_3":
                    required.add(f"layer_{li:02d}/probe_k.bin")
            if set(expected_hashes) != required:
                raise ValueError("activation hash manifest must cover every required file exactly")
            for relative in sorted(required):
                root = self.probe_root if relative.endswith("/probe_k.bin") else self.dir
                path = root / relative
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 << 20), b""):
                        digest.update(block)
                        self.activation["hash_verified_bytes"] += len(block)
                actual = digest.hexdigest()
                if actual != expected_hashes[relative]:
                    raise ValueError(f"activation hash mismatch: {relative}")
                self.activation["verified_hashes"][relative] = actual
            self.activation["hash_verification"] = "PASS"
            self.activation["hash_verification_seconds"] = time.perf_counter() - hash_started
        self.activation["seconds"] = time.perf_counter() - started
        self._active = None

    def _validate_layout(self, provenance):
        m = self.meta
        vn, heads, hd, layers = (int(m[x]) for x in ("v_token_num", "num_heads", "head_dim", "num_layers"))
        if min(vn, heads, hd, layers) <= 0 or int(m["chunk_size"]) != 64:
            raise ValueError("invalid canonical geometry or non-64 chunk size")
        if m["dtype"] not in ("float16", "float32"):
            raise ValueError("unsupported canonical storage dtype")
        if provenance and m["dtype"] != "float16":
            raise ValueError("frozen LLaVA SSD deployment requires float16 storage")
        if int(m["prefix_len"]) != int(m["v_token_start"]) + vn:
            raise ValueError("logical prefix geometry mismatch")
        if m.get("physical_layout", m.get("layout_method")) != "raster" or m.get("reordered", False) or m.get("order_is_per_layer", False):
            raise ValueError("only canonical/raster identity layout is supported")
        if m.get("order", list(range(vn))) != list(range(vn)):
            raise ValueError("nonidentity stored-to-original mapping")
        self.structural_ids = sorted(int(x) for x in m["newline_idx"])
        self.padding_ids = sorted(int(x) for x in m.get("padding_idx", []))
        excluded = set(self.structural_ids + self.padding_ids)
        if (len(excluded) != len(self.structural_ids) + len(self.padding_ids)
                or any(x < 0 or x >= vn for x in excluded)):
            raise ValueError("invalid or overlapping structural/padding IDs")
        if m.get("newline_stored", self.structural_ids) != self.structural_ids:
            raise ValueError("structural stored-to-original mapping mismatch")
        n_content = vn - len(excluded)
        if n_content <= 0 or int(m["n_spatial"]) != n_content:
            raise ValueError("invalid N_content")
        m["n_chunks_per_layer"] = (vn + 63) // 64
        if provenance:
            expected = {
                "layout_uses_dataset_question": False, "layout_uses_generated_answer": False,
                "calibration_questions": 0, "future_questions_used": 0,
                "turn1_normal_inference": True,
                "layout_source": "turn1_normal_inference_piggyback",
                "visual_kv_source": "turn1_captured_past_key_values",
                "visual_hidden_source": "same_turn1_decoder_layer0_input",
                "separate_vision_forward": False, "separate_prefix_forward": False,
                "separate_model_forward_for_visual_hidden": False,
                "capture_provenance_validated": True,
            }
            for name, expected_value in expected.items():
                if name not in m or m[name] != expected_value:
                    raise ValueError(f"unverified same-T1 capture provenance: {name}")
            hidden = m.get("hidden_capture", {})
            if hidden.get("capture_source") != "same_turn1_normal_multimodal_prefill" or hidden.get("visual_hidden_capture_count") != 1:
                raise ValueError("layer-0 visual embedding capture provenance mismatch")
        item = 2 if m["dtype"] == "float16" else 4
        for li in range(layers):
            for kind in ("k", "v"):
                if (self.dir / f"layer_{li:02d}" / f"{kind}.bin").stat().st_size != vn * heads * hd * item:
                    raise ValueError(f"canonical {kind} file size mismatch at layer {li}")
            if self.head_policy == "fixed_first_3":
                if self.probe_root == self.dir and int(m.get("probe_heads", 0)) != 3:
                    raise ValueError("Probe3 requires raw exactly-three-head sidecar")
                if (self.probe_root / f"layer_{li:02d}" / "probe_k.bin").stat().st_size != vn * 3 * hd * item:
                    raise ValueError(f"Probe3 sidecar size mismatch at layer {li}")
        n_sep = len(self.structural_ids)
        if (self.dir / "sep_kv.bin").stat().st_size != 2 * layers * n_sep * heads * hd * item:
            raise ValueError("structural sidecar size mismatch")

    def request(self):
        if self._active is not None:
            raise RuntimeError("one request per context; concurrent payload reuse is forbidden")
        result = CanonicalRequest(self)
        self._active = result
        return result

    def drop_cache(self):
        """OS page-cache conditioning outside TTFT, not NAND-cold assurance."""
        if self._active is not None:
            raise RuntimeError("page-cache conditioning must precede the timed request")
        paths = [self.dir / "sep_kv.bin"]
        for li in range(int(self.meta["num_layers"])):
            paths.extend(self.dir / f"layer_{li:02d}" / f"{kind}.bin" for kind in ("k", "v"))
            if self.head_policy == "fixed_first_3":
                paths.append(self.probe_root / f"layer_{li:02d}" / "probe_k.bin")
        for path in paths:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

    def close(self):
        if self._active is not None:
            self._active.close()
        self.sys_kv = None
        self.v_hidden = None


class CanonicalRequest:
    """Single-use request reader; only structural payload lives until close."""

    CATEGORIES = ("scoring_probe_k", "scoring_full_k", "selected_k", "selected_v", "structural")

    def __init__(self, context):
        self.context = context
        self.events = []
        meta = dict(context.meta)
        meta["probe_heads"] = 3
        self.reader = _EventReader(context.dir, meta, self.events, context.probe_root)
        self.scored = set()
        self.selected = set()
        self.layer_stats = []
        self.structural = None
        self.closed = False

    def __enter__(self):
        if self.closed:
            raise RuntimeError("request reader is single-use")
        return self

    def __exit__(self, *exc):
        self.close()

    def _check_layer(self, layer):
        if self.closed or not 0 <= int(layer) < int(self.context.meta["num_layers"]):
            raise ValueError("closed request or invalid layer")

    def prepare_structural(self):
        if self.closed:
            raise RuntimeError("closed request")
        if self.structural is not None:
            return self.structural
        m = self.context.meta
        shape = (2, int(m["num_layers"]), len(self.context.structural_ids), int(m["num_heads"]), int(m["head_dim"]))
        dtype = np.float16 if m["dtype"] == "float16" else np.float32
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if expected:
            fd = os.open(self.context.dir / "sep_kv.bin", os.O_RDONLY)
            try:
                started = time.perf_counter()
                blob = os.pread(fd, expected, 0)
                elapsed = time.perf_counter() - started
            finally:
                os.close(fd)
            self.events.append({"layer": None, "file": "sep_kv.bin", "category": "structural", "offset": 0,
                                "requested_bytes": expected, "returned_bytes": len(blob), "preads": 1, "seconds": elapsed})
            if len(blob) != expected:
                raise IOError("short structural sidecar read")
            self.structural = torch.from_numpy(np.frombuffer(blob, dtype=dtype).copy()).reshape(shape)
        else:
            self.structural = torch.empty(shape, dtype=torch.float16 if m["dtype"] == "float16" else torch.float32)
        return self.structural

    def read_scoring_keys(self, layer):
        self._check_layer(layer)
        if layer in self.scored:
            raise RuntimeError("scoring K may only be read once per layer/prefill")
        self.scored.add(layer)
        if self.context.head_policy == "all":
            self.reader.category = "scoring_full_k"
            return self.reader.read_full(layer, "k")
        self.reader.category = "scoring_probe_k"
        return self.reader.read_probe(layer)

    def read_selected(self, layer, plan, scoring_keys):
        self._check_layer(layer)
        if layer not in self.scored or layer in self.selected:
            raise RuntimeError("selection requires exactly one preceding scoring read")
        ctx, m = self.context, self.context.meta
        expected_plan = canonical_chunk_plan(plan["selected_tokens"], int(m["v_token_num"]), ctx.structural_ids, ctx.padding_ids)
        if plan != expected_plan:
            raise ValueError("selection plan differs from independently recomputed canonical mapping")
        expected_shape = (int(m["v_token_num"]), len(ctx.head_ids), int(m["head_dim"]))
        expected_dtype = torch.float16 if m["dtype"] == "float16" else torch.float32
        if tuple(scoring_keys.shape) != expected_shape or scoring_keys.dtype != expected_dtype or scoring_keys.device.type != "cpu":
            raise ValueError("scoring K must preserve native token-major CPU storage geometry")
        self.selected.add(layer)
        structural = self.prepare_structural()
        rows = torch.tensor(plan["keep_tokens"], dtype=torch.long)
        heads, hd = int(m["num_heads"]), int(m["head_dim"])
        key_out = torch.empty((len(rows), heads, hd), dtype=expected_dtype)
        val_out = torch.empty_like(key_out)
        self.reader.category = "selected_v"
        vrows, values = self.reader.read_chunks(layer, "v", plan["selected_chunks"])
        selected_rows = torch.tensor(plan["selected_tokens"], dtype=torch.long)
        physical_selected = torch.searchsorted(vrows, selected_rows)
        output_selected = torch.searchsorted(rows, selected_rows)
        structural_rows = torch.tensor(ctx.structural_ids, dtype=torch.long)
        output_structural = torch.searchsorted(rows, structural_rows)
        if ctx.head_policy == "all":
            # Logical gather only. No selected-K disk access or full-V fallback.
            key_out.copy_(scoring_keys[rows])
        else:
            self.reader.category = "selected_k"
            krows, keys = self.reader.read_chunks(layer, "k", plan["selected_chunks"])
            if not torch.equal(krows, vrows):
                raise RuntimeError("selected K/V physical rows disagree")
            key_out[output_selected] = keys.index_select(0, physical_selected)
        val_out[output_selected] = values.index_select(0, physical_selected)
        if ctx.structural_ids:
            if ctx.head_policy != "all":
                key_out[output_structural] = structural[0, layer]
            val_out[output_structural] = structural[1, layer]
        item = key_out.element_size()
        chunk_rows = len(plan["read_rows"])
        duplicate_probe = chunk_rows * 3 * hd if ctx.head_policy == "fixed_first_3" else 0
        duplicate_structural = (len(ctx.structural_ids) * heads * hd if ctx.head_policy == "all" else
                                (len(ctx.structural_ids) * 3 + plan["read_structural_rows"] * (heads - 3)) * hd)
        stats = {"layer": int(layer), "head_ids": list(ctx.head_ids), **plan,
                 "logical_reused_k_bytes": len(rows) * heads * hd * item if ctx.head_policy == "all" else 0,
                 "duplicated_probe_chunk_k_elements": duplicate_probe,
                 "duplicated_structural_k_elements": duplicate_structural,
                 "duplicated_k_elements": duplicate_probe + duplicate_structural,
                 "duplicated_k_bytes": (duplicate_probe + duplicate_structural) * item,
                 "duplicated_structural_v_bytes": plan["read_structural_rows"] * heads * hd * item,
                 "selected_k_read_bytes": 0 if ctx.head_policy == "all" else chunk_rows * heads * hd * item,
                 "selected_v_read_bytes": chunk_rows * heads * hd * item}
        self.layer_stats.append(stats)
        return LayerPayload(rows, key_out, val_out, stats)

    def summary(self):
        categories = {name: {"bytes": 0, "preads": 0, "seconds": 0.0} for name in self.CATEGORIES}
        for event in self.events:
            target = categories[event["category"]]
            target["bytes"] += event["returned_bytes"]
            target["preads"] += event["preads"]
            target["seconds"] += event["seconds"]
        return {"bytes": sum(x["bytes"] for x in categories.values()),
                "preads": sum(x["preads"] for x in categories.values()),
                "seconds": sum(x["seconds"] for x in categories.values()),
                "per_kind": categories, "events": list(self.events),
                "logical_reused_k_bytes": sum(x["logical_reused_k_bytes"] for x in self.layer_stats),
                "duplicated_k_bytes": sum(x["duplicated_k_bytes"] for x in self.layer_stats),
                "duplicated_structural_v_bytes": sum(x["duplicated_structural_v_bytes"] for x in self.layer_stats),
                "layers": list(self.layer_stats)}

    def close(self):
        if not self.closed:
            self.reader.close()
            self.structural = None
            self.closed = True
            if self.context._active is self:
                self.context._active = None

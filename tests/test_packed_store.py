"""CPU contracts for the PackedPrefixStore (contract v2, gates G6/G7 on fixtures).

Real FP16 files on the 2-layer fixture of test_llava_kv25: the packed file's
per-layer slices are bit-identical to the unpacked prefix25 / layerwise
writers, the single-preadv serving path restores the same cache rows and
masks as the unpacked selector, the read range does not depend on the
question, and tampered metadata / files fail closed.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mmimpress.layer_budget import budget_document, layer_keep_counts
from mmimpress.packed_serve import (PackedImageContext, PackedPrefixSelector,
                                    PackedReadTrace)
from mmimpress.packed_store import (PACKED_NAME, PackedReader, packed_file_sha256,
                                    packed_layout, validate_packed_meta,
                                    write_packed_store)
from mmimpress.prefix25 import seal_integrity, verify_integrity
from mmimpress.serve import BIAS, CVPR25ChunkSelector, ImageContext
from mmimpress.store import write_image_store
from test_llava_kv25 import _fixture as lfixture

CASES = (1, 63, 64, 65, 127, 128, 129, 256, 349, 2200)
RATIOS = {"U": [0.25, 0.25], "T": [0.35, 0.15]}


def _stores(root, n, alloc_id):
    meta_a, rank, layers = lfixture(root / "a", n)
    extra = {key: meta_a[key] for key in [
        "physical_layout", "layout_method", "layout_source", "visual_kv_source",
        "layout_uses_dataset_question", "llm_used_for_layout_scoring",
        "calibration_questions", "global_order_all_layers", "separator_tail",
        "separator_policy", "permutation_sha256", "inverse_permutation_sha256"]}
    doc = budget_document(alloc_id, RATIOS[alloc_id], model="fixture", source={}, num_layers=2)
    common = dict(extra=extra, stored_to_original=meta_a["order"], separator_sidecar=True, probe_heads=0)
    args = (layers, meta_a["v_token_start"], meta_a["v_token_num"], meta_a["prefix_input_ids"], meta_a["newline_idx"])
    if alloc_id == "U":
        u = write_image_store(root / "b", *args, storage_policy="prefix25", **common)
    else:
        u = write_image_store(root / "b", *args, storage_policy="prefix25_layerwise", layer_budget=doc, **common)
    timing = {}
    p = write_packed_store(root / "p", *args, storage_policy="prefix25_packed", layer_budget=doc, timing_out=timing, **common)
    for name in ("b", "p"):
        (root / name / "visionzip_layout.pt").write_bytes((root / "a" / "visionzip_layout.pt").read_bytes())
        seal_integrity(root / name)
    return meta_a, rank, layers, u, p, doc, timing


class PackedBytes(unittest.TestCase):
    def test_slices_bit_identical_to_unpacked_writers(self):
        for alloc in ("U", "T"):
            for n in CASES:
                with self.subTest(alloc=alloc, n=n), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    meta_a, rank, layers, u, p, doc, timing = _stores(root, n, alloc)
                    k = layer_keep_counts(RATIOS[alloc], n, num_layers=2)
                    self.assertEqual(p["layer_keep_counts"], k)
                    validate_packed_meta(p)
                    blob = (root / "p" / PACKED_NAME).read_bytes()
                    self.assertEqual(len(blob), p["packed_total_bytes"])
                    self.assertEqual(packed_file_sha256(root / "p" / PACKED_NAME), p["packed_sha256"])
                    row_bytes = 2 * 2 * 2
                    total = 0
                    for li in range(2):
                        for kind in ("k", "v"):
                            ref = (root / "b" / f"layer_{li:02d}/{kind}.bin").read_bytes()
                            off = p["packed_layout"][li][f"{kind}_offset"]
                            self.assertEqual(len(ref), k[li] * row_bytes)
                            self.assertEqual(blob[off:off + len(ref)], ref)
                            total += len(ref)
                    sep = (root / "b" / "sep_kv.bin").read_bytes()
                    self.assertEqual(blob[p["packed_sep_offset"]:], sep)
                    self.assertEqual(total + len(sep), len(blob))
                    self.assertEqual((root / "b" / "sys_kv.pt").read_bytes(), (root / "p" / "sys_kv.pt").read_bytes())
                    # shared metadata agrees with the unpacked writer
                    skip = {"format", "storage_policy", "layout", "order_semantics", "rounding_policy", "stored_row_to_original", "bytes_visual_kv"}
                    for key in set(u) & set(p) - skip:
                        self.assertEqual(u[key], p[key], key)
                    self.assertEqual(p["bytes_visual_kv"], sum(2 * kk * row_bytes for kk in k))
                    self.assertEqual(p["bytes_separator_sidecar"], len(sep))
                    self.assertIn("packed_sha256_ms", timing)
                    self.assertEqual(p["packed_writer_timing_ms"]["packed_sha256_ms"], timing["packed_sha256_ms"])
                    self.assertNotIn("payload_rows", p)

    def test_layout_table(self):
        table, sep_off, sep_bytes, total = packed_layout([3, 0, 5], 8, 3, 2)
        self.assertEqual([t["k_offset"] for t in table], [0, 48, 48])
        self.assertEqual([t["v_offset"] for t in table], [24, 48, 88])
        self.assertEqual(sep_off, 128)
        self.assertEqual(sep_bytes, 2 * 3 * 2 * 8)
        self.assertEqual(total, 128 + sep_bytes)
        with self.assertRaises(ValueError):
            packed_layout([3, -1], 8, 2, 1)


class PackedServing(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_single_preadv_restores_unpacked_cache_and_masks(self):
        runner = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
        for alloc in ("U", "T"):
            for n in CASES:
                with self.subTest(alloc=alloc, n=n), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    meta_a, rank, layers, u, p, doc, _ = _stores(root, n, alloc)
                    k = layer_keep_counts(RATIOS[alloc], n, num_layers=2)
                    # unpacked reference
                    ctx_u = ImageContext(root / "b", "cpu", drop_cache=False, require_v_hidden=False)
                    ctx_u.validate_visual_kv_layout()
                    cache_u = ctx_u.cache.new_request()
                    sel_u = CVPR25ChunkSelector(runner, ctx_u, None, .25, "prefix", sep_policy="sidecar", budget_unit="visual_kv")
                    with patch("torch.cuda.synchronize"):
                        sel_u.prepare(None)
                    bias_u = [BIAS[li].clone() for li in range(2)]
                    st_u = sel_u.stats()
                    BIAS.clear()
                    # packed, two questions and both copy modes
                    ctx_p = PackedImageContext(root / "p", "cpu", pin=False)
                    ctx_p.validate_packed_layout()
                    try:
                        for directcopy in (False, True):
                            for _ in range(2):
                                cache_p = ctx_p.cache.new_request()
                                sel_p = PackedPrefixSelector(runner, ctx_p, .25, directcopy=directcopy)
                                with patch("torch.cuda.synchronize"), PackedReadTrace(root / "p") as trace:
                                    sel_p.prepare()
                                s = trace.summary()
                                self.assertEqual(s["preads"], 1)
                                self.assertEqual(s["preadv_calls"], 1)
                                self.assertEqual(s["pread_calls"], 0)
                                self.assertEqual(s["bytes"], p["packed_total_bytes"])
                                self.assertEqual(s["calls"][0]["offset"], 0)
                                self.assertEqual(s["calls"][0]["requested"], s["calls"][0]["returned"])
                                self.assertEqual(sel_p.io.bytes, s["bytes"])
                                self.assertEqual(sel_p.io.preads, 1)
                                st_p = sel_p.stats()
                                self.assertEqual(st_p["k_per_layer"], k)
                                self.assertEqual(st_p["keep_count_per_layer"], st_u["keep_count_per_layer"])
                                self.assertEqual(st_p["selected_original_ids"], st_u["selected_original_ids"])
                                self.assertEqual(st_p["selected_original_ids"], rank[:max(k)])
                                self.assertEqual(st_p["read_calls"], 1)
                                for li in range(2):
                                    self.assertTrue(torch.equal(BIAS[li], bias_u[li]))
                                    for kind in ("keys", "values"):
                                        self.assertTrue(torch.equal(getattr(cache_p.layers[li], kind), getattr(cache_u.layers[li], kind)))
                                BIAS.clear()
                        with self.assertRaises(ValueError):
                            PackedPrefixSelector(runner, ctx_p, .2)
                    finally:
                        ctx_p.close(); ctx_u.close()

    def test_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta_a, rank, layers, u, p, doc, _ = _stores(root, 349, "T")
            k = p["layer_keep_counts"]
            for field, value in [("store_format", "x"), ("packed_file", "kv.bin"), ("packed_total_bytes", p["packed_total_bytes"] + 1),
                                 ("packed_sep_offset", 0), ("layer_keep_counts", [k[0] + 1, k[1] - 1]), ("layer_budget_sha256", "0" * 64),
                                 ("packed_layout", p["packed_layout"][::-1]), ("packed_sha256", "ab"), ("storage_policy", "prefix25_layerwise"),
                                 ("format", "llava_fp16_importance_prefix25_layerwise_v1"), ("payload_rows", k[0])]:
                bad = copy.deepcopy(p); bad[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_packed_meta(bad)
            # truncated file -> integrity and reader fail
            f = root / "p" / PACKED_NAME
            blob = f.read_bytes()
            f.write_bytes(blob[:-2])
            with self.assertRaises(ValueError):
                verify_integrity(root / "p")
            with self.assertRaises(ValueError):
                PackedReader(root / "p", p, pin=False)
            f.write_bytes(blob)
            # a store without integrity envelope is refused
            (root / "p" / "integrity.json").unlink()
            with self.assertRaises(ValueError):
                PackedImageContext(root / "p", "cpu", pin=False)
            # writer refusals
            args = (layers, meta_a["v_token_start"], meta_a["v_token_num"], meta_a["prefix_input_ids"], meta_a["newline_idx"])
            common = dict(stored_to_original=meta_a["order"], separator_sidecar=True, probe_heads=0)
            with self.assertRaises(ValueError):
                write_packed_store(root / "d1", *args, **common)                                   # no vector
            with self.assertRaises(ValueError):
                write_packed_store(root / "d2", *args, storage_policy="prefix25", layer_budget=doc, **common)
            wrong = dict(doc, layer_budget_sha256="0" * 64)
            with self.assertRaises(ValueError):
                write_packed_store(root / "d3", *args, layer_budget=wrong, **common)
            with self.assertRaises(ValueError):
                write_packed_store(root / "d4", *args, layer_budget=doc, stored_to_original=meta_a["order"], separator_sidecar=False, probe_heads=0)


if __name__ == "__main__":
    unittest.main()

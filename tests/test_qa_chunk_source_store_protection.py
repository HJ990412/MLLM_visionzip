"""CPU-only tests for QA-Chunk25 read-only source-store protection."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/51_protect_qa_chunk_source_stores.py"
SPEC = importlib.util.spec_from_file_location(
    "qa_chunk_source_store_protection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROTECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTECT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceStoreProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.store = self.root / "source_store"
        self.run = self.root / "source_run"
        self.results = self.root / "source_results"
        self.index = self.root / "index.json"
        self.out = self.root / "new_run"
        self.run.mkdir()
        self.results.mkdir()
        self.out.mkdir()
        (self.store / "raster").mkdir(parents=True)
        (self.store / "image_only").mkdir()
        self.image_ids = ["image_a", "image_b"]
        index = []
        for image_number, image_id in enumerate(self.image_ids):
            index.append({
                "image_id": image_id,
                "image_path": f"unused/{image_id}.jpg",
                "questions": [
                    {"question_id": f"q{image_number}_{question}"}
                    for question in range(10)
                ],
            })
        self.index.write_text(json.dumps(index), encoding="utf-8")
        identity = PROTECT._workload_identity(self.index, 4, 6)
        self.expected_index_sha = identity["index_sha256"]
        self.expected_workload_sha = identity["workload_sha256"]

        persistence = []
        for image_number, image_id in enumerate(self.image_ids):
            qa = self._make_store(image_id, image_number, "qa")
            ours = self._make_store(image_id, image_number, "ours")
            persistence.append({"image_id": image_id, "qa": qa,
                                "ours": ours})
        persistence_path = self.run / "persistence.jsonl"
        persistence_path.write_text("".join(
            json.dumps(row) + "\n" for row in persistence), encoding="utf-8")
        config = {
            "schema_version": "fixture",
            "status": "complete",
            "dataset": "gqa",
            "index_sha256": self.expected_index_sha,
            "full_workload_sha256": self.expected_workload_sha,
            "selected_workload_sha256": self.expected_workload_sha,
            "full_images": 2,
            "selected_images": 2,
            "n_images": 2,
            "full_questions": 12,
            "selected_questions": 12,
            "n_questions": 12,
            "model": "fixture-model",
            "chunk_size": 2,
            "probe_heads": 1,
            "skip": 4,
            "questions_per_image": 6,
            "future_question_leakage": 0,
            "method_keys": list(PROTECT.EXPECTED_METHOD_KEYS),
            "run_dir": str(self.run),
            "store_dir": str(self.store),
            "results_dir": str(self.results),
        }
        config_path = self.run / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        artifacts = {
            "schema_version": "fixture",
            "run_dir": str(self.run),
            "store_dir": str(self.store),
            "files_sha256": {
                "config.json": _sha(config_path),
                "persistence.jsonl": _sha(persistence_path),
            },
        }
        (self.results / "run_artifacts.json").write_text(
            json.dumps(artifacts), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _make_store(self, image_id: str, image_number: int,
                    side: str) -> dict:
        layout = "raster" if side == "qa" else "image_only"
        path = self.store / layout / image_id
        path.mkdir()
        for layer in range(2):
            directory = path / f"layer_{layer:02d}"
            directory.mkdir()
            marker = bytes([image_number + layer + 1])
            (directory / "k.bin").write_bytes(marker * 8)
            (directory / "v.bin").write_bytes(marker * 8)
            (directory / "probe_k.bin").write_bytes(
                marker * (8 if side == "qa" else 0))
        # Large enough that a middle-byte mutation is outside the writer's
        # 4 KiB head/tail sample, allowing the optional full-hash test below.
        (path / "sys_kv.pt").write_bytes(b"s" * 12000)
        (path / "sep_kv.bin").write_bytes(b"e" * 8)
        if side == "qa":
            (path / "v_hidden.pt").write_bytes(b"h" * 17)
        else:
            (path / "visionzip_layout.pt").write_bytes(b"l" * 19)

        common = {
            "v_token_start": 1,
            "v_token_num": 4,
            "prefix_len": 5,
            "num_layers": 2,
            "num_heads": 1,
            "head_dim": 1,
            "dtype": "float16",
            "chunk_size": 2,
            "n_chunks_per_layer": 2,
            "newline_idx": [3],
            "n_spatial": 3,
            "prefix_input_ids": [1, 32000, 32000, 32000, 32000],
            "bytes_visual_kv": 32,
            "bytes_separator_sidecar": 8,
            "image_id": image_id,
            "model": "fixture-model",
            "dataset": "gqa",
            "source_turn_id": 1,
            "layout_source": "turn1_normal_inference_piggyback",
            "visual_kv_source": "turn1_captured_past_key_values",
            "turn1_normal_inference": True,
            "capture_provenance_validated": True,
            "image_input_sha256": hashlib.sha256(
                image_id.encode("utf-8")).hexdigest(),
        }
        if side == "qa":
            meta = {
                **common,
                "probe_heads": 1,
                "bytes_probe_sidecar": 16,
                "physical_layout": "raster",
                "layout_method": "raster",
                "reordered": False,
                "order_is_per_layer": False,
                "layout_uses_dataset_question": False,
                "layout_uses_generated_answer": False,
                "calibration_questions": 0,
                "future_questions_used": 0,
                "qa_select_compatible": True,
                "probe_heads_required_for_serving": 1,
                "visual_hidden_source": "same_turn1_decoder_layer0_input",
                "separate_vision_forward": False,
                "separate_prefix_forward": False,
                "separate_model_forward_for_visual_hidden": False,
                "hidden_capture": {
                    "capture_source": "same_turn1_normal_multimodal_prefill",
                    "visual_hidden_capture_count": 1,
                },
            }
        else:
            permutation = [1, 0, 2, 3]
            meta = {
                **common,
                "probe_heads": 0,
                "bytes_probe_sidecar": 0,
                "physical_layout": "visionzip_image_only",
                "layout_method": "visionzip_image_only",
                "reordered": True,
                "order": permutation,
                "order_is_per_layer": False,
                "newline_stored": [3],
                "global_order_all_layers": True,
                "layout_uses_dataset_question": False,
                "llm_used_for_layout_scoring": False,
                "calibration_questions": 0,
                "future_questions_used_for_layout": 0,
                "probe_heads_required_for_serving": 0,
                "separator_tail": True,
                "permutation_sha256": hashlib.sha256(b"permutation").hexdigest(),
                "inverse_permutation_sha256": hashlib.sha256(b"inverse").hexdigest(),
            }
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        sizes = {
            file.relative_to(path).as_posix(): file.stat().st_size
            for file in sorted(item for item in path.rglob("*")
                               if item.is_file())
        }
        small = ["meta.json", ("v_hidden.pt" if side == "qa"
                               else "visionzip_layout.pt")]
        return {
            "store_dir": str(path),
            "image_id": image_id,
            "meta": meta,
            "file_sizes": sizes,
            "hashes": {
                "files_sha256": {name: _sha(path / name) for name in small},
                "tree_sha256": None,
                "prefix_kv_sample_sha256": PROTECT._sampled_store_sha256(
                    path, sizes),
                "full_integrity_hash": False,
            },
            "integrity": {"ok": True, "error": None},
        }

    def _kwargs(self, full_payload_hash: bool = False) -> dict:
        return {
            "source_store": self.store,
            "source_run": self.run,
            "source_results": self.results,
            "index": self.index,
            "expected_index_sha256": self.expected_index_sha,
            "expected_workload_sha256": self.expected_workload_sha,
            "expected_images": 2,
            "expected_questions": 12,
            "expected_model": "fixture-model",
            "expected_layers": 2,
            "expected_heads": 1,
            "expected_head_dim": 1,
            "expected_probe_heads": 1,
            "expected_chunk_size": 2,
            "full_payload_hash": full_payload_hash,
        }

    def test_fast_snapshot_validates_both_layouts_and_all_images(self):
        snapshot = PROTECT.build_snapshot(**self._kwargs())
        self.assertEqual(snapshot["hash_mode"],
                         "writer_head_tail_4096_every_file_plus_full_controls")
        self.assertFalse(snapshot["full_payload_hash"])
        self.assertIsNone(snapshot["source_store_tree_sha256"])
        self.assertEqual(set(snapshot["stores"]), set(self.image_ids))
        self.assertEqual(snapshot["source_file_count"], 40)
        self.assertEqual(
            {row["raster"]["physical_layout"]
             for row in snapshot["stores"].values()}, {"raster"})
        self.assertEqual(
            {row["image_only"]["physical_layout"]
             for row in snapshot["stores"].values()},
            {"visionzip_image_only"})
        for row in snapshot["stores"].values():
            self.assertEqual(row["raster"]["image_input_sha256"],
                             row["image_only"]["image_input_sha256"])

    def test_fast_before_after_verification_is_exclusive_and_unchanged(self):
        manifest = self.out / "source_stores_before.json"
        validation = self.out / "source_stores_validation.json"
        path, before = PROTECT.record_before(manifest, **self._kwargs())
        self.assertEqual(path, manifest)
        with self.assertRaises(FileExistsError):
            PROTECT.record_before(manifest, **self._kwargs())
        validation_path, report = PROTECT.verify_after(manifest, validation)
        self.assertEqual(validation_path, validation)
        self.assertTrue(report["passed"])
        self.assertTrue(report["read_only_source_reuse_validated"])
        self.assertFalse(report["full_payload_hash"])
        self.assertEqual(before["manifest_sha256"],
                         report["after_manifest_sha256"])

    def test_optional_full_hash_detects_unsampled_middle_byte_change(self):
        manifest = self.out / "source_stores_full_before.json"
        validation = self.out / "source_stores_full_validation.json"
        _, before = PROTECT.record_before(
            manifest, **self._kwargs(full_payload_hash=True))
        self.assertTrue(before["full_payload_hash"])
        self.assertIsNotNone(before["source_store_tree_sha256"])
        target = self.store / "raster/image_a/sys_kv.pt"
        with target.open("r+b") as handle:
            handle.seek(6000)
            handle.write(b"X")
        with self.assertRaises(PROTECT.SourceStoreProtectionError) as caught:
            PROTECT.verify_after(manifest, validation)
        self.assertIsNotNone(caught.exception.report)
        report = caught.exception.report
        assert report is not None
        self.assertFalse(report["passed"])
        self.assertIn("raster/image_a/sys_kv.pt", report["changed_paths"])
        self.assertTrue(validation.is_file())

    def test_wrong_raster_layout_fails_closed(self):
        meta_path = self.store / "raster/image_a/meta.json"
        meta = json.loads(meta_path.read_text())
        meta["physical_layout"] = "visionzip_image_only"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(
                PROTECT.SourceStoreProtectionError,
                "persisted and on-disk qa metadata differ"):
            PROTECT.build_snapshot(**self._kwargs())

    def test_missing_image_store_and_workload_hash_fail_closed(self):
        moved = self.store / "raster/image_b.moved"
        (self.store / "raster/image_b").rename(moved)
        with self.assertRaisesRegex(PROTECT.SourceStoreProtectionError,
                                    "image inventory"):
            PROTECT.build_snapshot(**self._kwargs())
        moved.rename(self.store / "raster/image_b")
        bad = self._kwargs()
        bad["expected_workload_sha256"] = "0" * 64
        with self.assertRaisesRegex(PROTECT.SourceStoreProtectionError,
                                    "workload SHA256"):
            PROTECT.build_snapshot(**bad)


if __name__ == "__main__":
    unittest.main()

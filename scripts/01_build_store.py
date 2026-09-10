"""Build the disk-resident prefix-KV store: one image == one prefix.

For every image in data/index.json this runs a single prefill over
[system tokens | expanded image block] -- deliberately WITHOUT any question, so
the cached KV is a prefix of every request about that image -- and writes it
token-major (see mmimpress/store.py).  The pre-layer-0 hidden states of the
visual block are saved too: they are question-independent, and having them on
disk lets the serving path pick SparseVLM raters without ever running the
vision tower.

  python scripts/01_build_store.py [--limit N]
"""
import argparse
import gc
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import CHUNK_SIZE, PROJECT_ROOT, STORE_DIR
from mmimpress.cvpr25 import (anyres_token_scores, clip_cls_patch_saliency,
                              permutation_sha256, visionzip_repack_order)
from mmimpress.dataset import load_index
from mmimpress.model import LlavaRunner, cache_layers
from mmimpress.reorder import mapping_from_perm
from mmimpress.serve import suffix_ids_for
from mmimpress.store import write_image_store


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_pixels(tensor):
    """Normalise AutoProcessor and image_processor batch conventions."""
    return tensor[0] if tensor.dim() == 5 and tensor.shape[0] == 1 else tensor


@torch.no_grad()
def build_one(runner, entry, out_root: Path, layout="raster",
              separator_sidecar=False):
    ingest_t0 = time.perf_counter()
    img = Image.open(PROJECT_ROOT / entry["image_path"]).convert("RGB")
    q0 = entry["questions"][0]["question"]
    enc = runner.encode(img, q0)
    ids = enc["input_ids"][0]
    v0, vn = runner.visual_span(ids)
    prefix_len = v0 + vn
    base, hi_h, hi_w, nl = runner.anyres_layout(
        enc["image_sizes"][0].tolist(), vn)

    # the question-side ids the server will reconstruct must line up with what
    # the processor produced here, or the stored prefix is not a real prefix
    suf = suffix_ids_for(runner, q0)
    assert torch.equal(ids[prefix_len:], suf.to(ids.device)), \
        "suffix reconstruction differs from the processor output"

    # Freeze image-only layout before the LLM prefix forward.  The scorer API
    # has no text/question argument and calls only the vision tower.
    stored_to_original = None
    token_score = None
    saliency_ms = permutation_ms = 0.0
    image_inputs_match_prompt_processor = None
    image_input_sha256 = None
    if layout == "visionzip":
        image_enc = runner.image_inputs(img)
        px_image = _canonical_pixels(image_enc["pixel_values"])
        px_prompt = _canonical_pixels(enc["pixel_values"])
        sizes_image = image_enc["image_sizes"].reshape(-1)
        sizes_prompt = enc["image_sizes"].reshape(-1)
        image_inputs_match_prompt_processor = bool(
            torch.equal(px_image, px_prompt)
            and torch.equal(sizes_image, sizes_prompt))
        assert image_inputs_match_prompt_processor, \
            "text-free image preprocessing differs from prompt preprocessing"
        image_input_sha256 = hashlib.sha256(
            px_image.contiguous().numpy().tobytes()
            + sizes_image.contiguous().numpy().tobytes()).hexdigest()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        per_sub = clip_cls_patch_saliency(runner,
                                          image_enc["pixel_values"])
        token_score = anyres_token_scores(
            runner, per_sub, image_enc["image_sizes"][0].tolist(), vn)
        torch.cuda.synchronize()
        saliency_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        stored_to_original = visionzip_repack_order(token_score, nl)
        permutation_ms = (time.perf_counter() - t0) * 1e3
        assert stored_to_original[-len(nl):] == list(nl)

    box = {}

    def grab(module, args, kwargs):
        if "h" not in box:
            box["h"] = (args[0] if args else kwargs["hidden_states"]).detach()

    hook = runner.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    torch.cuda.synchronize()
    prefix_t0 = time.perf_counter()
    try:
        out = runner.prefix_forward(enc, prefix_len)
    finally:
        hook.remove()
    torch.cuda.synchronize()
    prefix_forward_ms = (time.perf_counter() - prefix_t0) * 1e3

    layers = cache_layers(out.past_key_values)
    assert layers[0][0].shape[2] == prefix_len, layers[0][0].shape

    out_dir = out_root / str(entry["image_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    torch.save(box["h"][0, v0:prefix_len].to(torch.float16).cpu(),
               out_dir / "v_hidden.pt")
    v_hidden_write_ms = (time.perf_counter() - t0) * 1e3
    image_path = PROJECT_ROOT / entry["image_path"]
    method = "visionzip_image_only" if layout == "visionzip" else "raster"
    extra = {
        "image_id": entry["image_id"], "model": runner.model_id,
        "base_grid": base, "hires_grid": [hi_h, hi_w],
        "n_questions": len(entry["questions"]),
        "physical_layout": method,
        "layout_method": method,
        "layout_source": "fresh_model_forward",
        "composed_from_store": False,
        "layout_uses_dataset_question": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
        "layout_frozen_before_llm_prefix_forward": True,
        "global_order_all_layers": True,
        "separator_policy": ("stable_tail_plus_sidecar" if layout == "visionzip"
                             else "original_positions_plus_sidecar"),
        "separator_tail": layout == "visionzip",
        "source_image_sha256": _sha256(image_path),
        # q0 is used only to prove/cut the reusable prefix.  It is not passed
        # to image_inputs(), clip_cls_patch_saliency(), or the order helper.
        "dataset_question_used_for_prefix_boundary_only": True,
        "layout_question_argument_available": False,
        "image_inputs_match_prompt_processor":
            image_inputs_match_prompt_processor,
    }
    if stored_to_original is not None:
        extra.update({
            "permutation_sha256": permutation_sha256(stored_to_original),
            "inverse_permutation_sha256": permutation_sha256(
                mapping_from_perm(stored_to_original)),
        })
    writer_timing = {}
    meta = write_image_store(
        out_dir, layers, v0, vn, ids[:prefix_len].tolist(), nl,
        chunk_size=CHUNK_SIZE,
        extra=extra, stored_to_original=stored_to_original,
        separator_sidecar=separator_sidecar or layout == "visionzip",
        timing_out=writer_timing)

    layout_metadata_bytes = 0
    if layout == "visionzip":
        layout_artifact = {
            "schema_version": 1,
            "image_id": str(entry["image_id"]),
            "importance_source":
                "vision_encoder_penultimate_cls_to_patch_attention_head_sum",
            "token_score_original": token_score,
            "stored_to_original": torch.tensor(stored_to_original,
                                                dtype=torch.int32),
            "original_to_stored": torch.tensor(
                mapping_from_perm(stored_to_original), dtype=torch.int32),
            "newline_original": torch.tensor(nl, dtype=torch.int32),
            "layout_uses_dataset_question": False,
            "llm_used_for_layout_scoring": False,
            "calibration_questions": 0,
            "image_input_sha256": image_input_sha256,
        }
        t0 = time.perf_counter()
        torch.save(layout_artifact, out_dir / "visionzip_layout.pt")
        writer_timing["ssd_write_ms"] += (time.perf_counter() - t0) * 1e3
        layout_metadata_bytes = (out_dir / "visionzip_layout.pt").stat().st_size

    total_ingestion_ms = (time.perf_counter() - ingest_t0) * 1e3
    profile = {
        "image_id": str(entry["image_id"]),
        "layout": method,
        "v_token_num": int(vn),
        "real_visual_tokens": int(vn - len(nl)),
        "separator_tokens": int(len(nl)),
        "n_chunks": int(meta["n_chunks_per_layer"]),
        "vision_saliency_ms": float(saliency_ms),
        "permutation_ms": float(permutation_ms),
        "prefix_forward_ms": float(prefix_forward_ms),
        "kv_materialize_ms": float(writer_timing["kv_materialize_ms"]),
        "kv_repack_ms": float(writer_timing["kv_repack_ms"]),
        "ssd_write_ms": float(writer_timing["ssd_write_ms"]
                              + v_hidden_write_ms),
        "total_ingestion_ms": float(total_ingestion_ms),
        "mapping_metadata_bytes": int(layout_metadata_bytes
                                      + (out_dir / "meta.json").stat().st_size),
        "visual_kv_bytes": int(meta["bytes_visual_kv"]),
        "probe_sidecar_bytes": int(meta["bytes_probe_sidecar"]),
        "separator_sidecar_bytes": int(meta["bytes_separator_sidecar"]),
        "permutation_sha256": meta.get("permutation_sha256"),
        "inverse_permutation_sha256": meta.get(
            "inverse_permutation_sha256"),
        "image_input_sha256": image_input_sha256,
        "image_inputs_match_prompt_processor":
            image_inputs_match_prompt_processor,
        "layout_uses_dataset_question": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
        "layout_frozen_before_llm_prefix_forward": True,
    }
    del out, layers, box
    return meta, profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(STORE_DIR))
    ap.add_argument("--index", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated image_ids to build (shard)")
    ap.add_argument("--layout", choices=("raster", "visionzip"),
                    default="raster",
                    help="physical Visual-KV layout written directly")
    ap.add_argument("--separator-sidecar", action="store_true",
                    help="write sep_kv.bin without building static.pt")
    ap.add_argument("--require-empty-output", action="store_true",
                    help="fail before model load unless --out is absent/empty")
    ap.add_argument("--profile-out", default=None,
                    help="write per-image ingestion timing/provenance JSON")
    args = ap.parse_args()

    index = load_index(args.index)
    if args.only:
        keep = set(args.only.split(","))
        index = [e for e in index if str(e["image_id"]) in keep]
    if args.limit:
        index = index[:args.limit]
    out_root = Path(args.out)
    if args.require_empty_output:
        assert out_root.resolve() != STORE_DIR.resolve(), \
            "refusing fresh/no-clobber mode on the canonical historical store"
        assert not out_root.is_symlink(), "refusing symlink output root"
        if out_root.exists():
            assert out_root.is_dir() and not any(out_root.iterdir()), \
                f"fresh output is not empty: {out_root}"
    if args.layout == "visionzip":
        assert out_root.resolve() != STORE_DIR.resolve(), \
            "VisionZip direct build must not overwrite the historical store"
    out_root.mkdir(parents=True, exist_ok=True)
    runner = LlavaRunner().load()

    total = 0
    profiles = []
    for i, e in enumerate(index):
        d = out_root / str(e["image_id"])
        if (d / "meta.json").exists():
            print(f"[{i+1}/{len(index)}] {e['image_id']}: exists, skip")
            continue
        t0 = time.time()
        m, profile = build_one(
            runner, e, out_root, layout=args.layout,
            separator_sidecar=args.separator_sidecar)
        profiles.append(profile)
        total += m["bytes_visual_kv"]
        print(f"[{i+1}/{len(index)}] {e['image_id']}: v_num={m['v_token_num']} "
              f"chunks={m['n_chunks_per_layer']} "
              f"{m['bytes_visual_kv']/1e9:.2f}GB+{m['bytes_probe_sidecar']/1e9:.2f} ({time.time()-t0:.1f}s)")
        if (i + 1) % 4 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    print(f"store: {total/1e9:.1f} GB visual KV in {out_root}")
    if args.profile_out:
        profile_out = Path(args.profile_out)
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "command": shlex.join([sys.executable, *sys.argv]),
            "layout": args.layout,
            "store": str(out_root.resolve()),
            "fresh_no_clobber": bool(args.require_empty_output),
            "layout_uses_dataset_question": False,
            "llm_used_for_layout_scoring": False,
            "calibration_questions": 0,
            "n_images": len(profiles),
            "rows": profiles,
        }
        profile_out.write_text(json.dumps(document, indent=1))
        print(f"profile: {profile_out}")


if __name__ == "__main__":
    main()

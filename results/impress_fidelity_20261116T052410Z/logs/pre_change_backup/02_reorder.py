"""KV reordering pass (IMPRESS 4.4.1) over the image prefix store.

Without this the store sits in raster order and every 64-token chunk is a thin
horizontal strip of the patch grid, so a scattered selection touches
essentially every chunk and selective loading reads MORE than a full load (the
probe sidecar and the fallback layers are pure overhead).  Repacking is what
turns identification into I/O savings.

Target orders:
  importance  IMPRESS 4.4.1 -- descending AVERAGE SparseVLM importance,
              accumulated over several questions about the same image, one
              order per layer (the consensus path applies one token set to all
              heads, so a per-layer order is what the read pattern wants).
              ``--reference-order-store`` preserves an existing calibrated
              normal-token order while moving separators to a matched tail.
  morton      query-independent Z-order of the patch grid, so a chunk is a
              compact spatial tile instead of a strip; needs no calibration
  visionzip   query-independent Vision Encoder CLS-to-patch saliency order;
              accepted only from an explicit fresh raster store
  raster      restore a temporary transformed store to original token order

Rewrites layer_*/k.bin and v.bin in place and records the permutation in
meta.json.  The serving path works in stored positions, so nothing has to be
un-permuted at request time.

  python scripts/02_reorder.py --order importance --calib-questions 4
"""
import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import PROJECT_ROOT, STORE_DIR
from mmimpress.cvpr25 import (anyres_token_scores, clip_cls_patch_saliency,
                              permutation_sha256, visionzip_repack_order)
from mmimpress.dataset import load_index
from mmimpress.reorder import llava_visual_order, mapping_from_perm
from mmimpress.store import (load_meta, write_separator_sidecar_from_store)

_NP = {"float16": np.float16, "float32": np.float32}


def current_order(meta, layer):
    """stored -> original for one layer of the CURRENT store.

    meta["order"] is a single list when every layer shares an order and a list
    of per-layer lists otherwise; a store that was never reordered has none.
    """
    o = meta.get("order")
    if not o:
        return list(range(meta["v_token_num"]))
    return o[layer] if meta.get("order_is_per_layer") else o


def rewrite_layer(ldir: Path, meta, perm_np):
    """Permute the token axis of the token-major (v_num, W, hd) blocks.

    The probe sidecar has to move with them or identification would score the
    wrong rows.
    """
    vn, hd = meta["v_token_num"], meta["head_dim"]
    dt = _NP[meta["dtype"]]
    for name, width in (("k", meta["num_heads"]), ("v", meta["num_heads"]),
                        ("probe_k", meta["probe_heads"])):
        p = ldir / f"{name}.bin"
        a = np.fromfile(p, dtype=dt).reshape(vn, width, hd)
        rewritten = np.ascontiguousarray(a[perm_np])
        tmp = p.with_name(p.name + ".reorder-tmp")
        if tmp.exists():
            raise FileExistsError(f"stale reorder temp file: {tmp}")
        rewritten.tofile(tmp)
        os.replace(tmp, p)


def commit(store_dir: Path, meta, perms, metadata_update=None,
           identity_layout=False):
    """perms: {layer -> permutation over CURRENT stored positions}."""
    per_layer_order, per_layer_nl = [], []
    for li in range(meta["num_layers"]):
        cur = current_order(meta, li)
        perm = perms[li]
        rewrite_layer(store_dir / f"layer_{li:02d}", meta, np.asarray(perm))
        new_order = [cur[p] for p in perm]            # stored -> original
        inv = mapping_from_perm(new_order)            # original -> stored
        per_layer_order.append(new_order)
        per_layer_nl.append([inv[i] for i in meta["newline_idx"]])

    same = all(o == per_layer_order[0] for o in per_layer_order)
    meta["order"] = per_layer_order[0] if same else per_layer_order
    meta["order_is_per_layer"] = not same
    # separators must be locatable per layer; when the order is shared this is
    # a single list, otherwise the selector needs the per-layer variant
    meta["newline_stored"] = (per_layer_nl[0] if same else per_layer_nl)
    meta["reordered"] = not identity_layout
    if metadata_update:
        meta.update(metadata_update)
    with open(store_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return same


def _is_identity_order(meta):
    vn, L = int(meta["v_token_num"]), int(meta["num_layers"])
    identity = list(range(vn))
    return all(current_order(meta, li) == identity for li in range(L))


def require_fresh_raster(meta, store_dir: Path):
    """Fail closed before the image-only prototype can compose layouts."""
    assert _is_identity_order(meta), \
        f"VisionZip prototype source is not raster identity: {store_dir}"
    assert not meta.get("reordered", False), \
        f"VisionZip prototype source says reordered: {store_dir}"
    layout = meta.get("physical_layout", meta.get("layout_method", "raster"))
    assert layout in (None, "raster"), \
        f"VisionZip prototype source layout is {layout!r}, not raster"
    assert meta.get("layout_source") in (None, "fresh_model_forward"), \
        f"VisionZip prototype source was not a fresh model forward: {store_dir}"
    assert not meta.get("composed_from_store", False), \
        f"VisionZip prototype source is already composed: {store_dir}"


def target_to_current_perms(meta, target_orders):
    """Convert target stored->original orders to current-store positions."""
    L = int(meta["num_layers"])
    targets = (target_orders if isinstance(target_orders[0], list)
               else [target_orders] * L)
    assert len(targets) == L
    answer = {}
    for li in range(L):
        pos = {int(original): stored
               for stored, original in enumerate(current_order(meta, li))}
        target = [int(i) for i in targets[li]]
        assert sorted(target) == list(range(meta["v_token_num"]))
        answer[li] = [pos[original] for original in target]
    return answer


def _file_sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=("importance", "morton", "visionzip",
                                        "raster"),
                    default="importance")
    ap.add_argument("--calib-questions", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--store", default=str(STORE_DIR))
    ap.add_argument("--index", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated image_ids")
    ap.add_argument("--separator-tail", action="store_true",
                    help="importance only: exclude separators from ranking and "
                         "append them in original order")
    ap.add_argument("--rebuild-separator-sidecar", action="store_true")
    ap.add_argument("--allow-composed-raster-target", action="store_true",
                    help="allow --order raster to restore a temporary store")
    ap.add_argument("--record-layout-provenance", action="store_true",
                    help="add explicit layout metadata for new experiments")
    ap.add_argument("--profile-out", default=None)
    ap.add_argument(
        "--reference-order-store", default=None,
        help=("importance+separator-tail only: reuse an existing calibrated "
              "stored->original order and move only separators to a stable "
              "tail, for an exact matched separator-policy control"))
    args = ap.parse_args()

    if args.separator_tail and args.order != "importance":
        ap.error("--separator-tail is valid only with --order importance")
    if args.reference_order_store and not (
            args.order == "importance" and args.separator_tail):
        ap.error("--reference-order-store requires importance + --separator-tail")
    index = load_index(args.index)
    if args.only:
        keep = set(args.only.split(","))
        index = [e for e in index if str(e["image_id"]) in keep]
    if args.limit:
        index = index[:args.limit]
    store = Path(args.store)

    if args.order in ("visionzip", "raster"):
        assert store.resolve() != STORE_DIR.resolve(), \
            f"refusing {args.order} rewrite of the protected canonical store"
    if args.order == "raster" and not args.allow_composed_raster_target:
        raise AssertionError(
            "--order raster is a temporary-store restore operation; pass "
            "--allow-composed-raster-target explicitly")

    reference_order_store = (Path(args.reference_order_store).resolve()
                             if args.reference_order_store else None)
    runner = server = None
    if (args.order == "visionzip"
            or (args.order == "importance" and reference_order_store is None)):
        from mmimpress.model import LlavaRunner
        runner = LlavaRunner().load()
        if args.order == "importance":
            from mmimpress.serve import Server
            server = Server(runner)

    profiles = []
    for i, e in enumerate(index):
        d = store / str(e["image_id"])
        if not (d / "meta.json").exists():
            continue
        meta = load_meta(d)
        t0 = time.perf_counter()
        before_meta_sha = _file_sha256(d / "meta.json")
        saliency_ms = 0.0
        token_score = None
        target_orders = None

        if args.order == "morton":
            base = meta["base_grid"]
            hi_h, hi_w = meta["hires_grid"]
            target_orders, _ = llava_visual_order(
                base, hi_h, hi_w, meta["newline_idx"], meta["v_token_num"])
            perms = target_to_current_perms(meta, target_orders)
        elif args.order == "raster":
            target_orders = list(range(meta["v_token_num"]))
            perms = target_to_current_perms(meta, target_orders)
        elif args.order == "visionzip":
            require_fresh_raster(meta, d)
            img = Image.open(PROJECT_ROOT / e["image_path"]).convert("RGB")
            image_enc = runner.image_inputs(img)
            torch.cuda.synchronize()
            score_t0 = time.perf_counter()
            per_sub = clip_cls_patch_saliency(
                runner, image_enc["pixel_values"])
            token_score = anyres_token_scores(
                runner, per_sub, image_enc["image_sizes"][0].tolist(),
                meta["v_token_num"])
            torch.cuda.synchronize()
            saliency_ms = (time.perf_counter() - score_t0) * 1e3
            target_orders = visionzip_repack_order(
                token_score, meta["newline_idx"])
            perms = target_to_current_perms(meta, target_orders)
        else:  # unchanged IMPRESS importance path, or exact reference reuse
            if reference_order_store is not None:
                ref_dir = reference_order_store / str(e["image_id"])
                ref_meta = load_meta(ref_dir)
                for key in ("v_token_num", "num_layers", "newline_idx",
                            "chunk_size", "num_heads", "head_dim"):
                    assert ref_meta[key] == meta[key], \
                        f"reference geometry mismatch {e['image_id']}/{key}"
                assert ref_meta.get("reordered") is True, \
                    f"reference store is not reordered: {ref_dir}"
                assert ref_meta.get("order_is_per_layer") is True, \
                    f"reference store is not calib-style per-layer: {ref_dir}"
                original_sep = sorted(int(x) for x in meta["newline_idx"])
                sep_set = set(original_sep)
                target_orders = []
                for li in range(meta["num_layers"]):
                    ref_order = [int(x) for x in current_order(ref_meta, li)]
                    assert sorted(ref_order) == list(range(meta["v_token_num"]))
                    target_orders.append(
                        [x for x in ref_order if x not in sep_set]
                        + original_sep)
                perms = target_to_current_perms(meta, target_orders)
            else:
                from mmimpress.serve import ImageContext, calibrate_image
                ctx = ImageContext(d, runner.model.device, drop_cache=False)
                qs = [q["question"]
                      for q in e["questions"][:args.calib_questions]]
                scores = calibrate_image(server, ctx, qs)  # (L, v_num)
                ctx.close()
                del ctx
                torch.cuda.empty_cache()
            if args.separator_tail and reference_order_store is None:
                targets = []
                original_sep = [int(x) for x in meta["newline_idx"]]
                for li in range(meta["num_layers"]):
                    cur = current_order(meta, li)
                    sep_current = set(
                        mapping_from_perm(cur)[orig] for orig in original_sep)
                    normal_current = [p for p in range(meta["v_token_num"])
                                      if p not in sep_current]
                    normal_t = torch.tensor(normal_current, dtype=torch.long)
                    ranked = normal_t[torch.argsort(
                        scores[li][normal_t], descending=True,
                        stable=True)].tolist()
                    targets.append([cur[p] for p in ranked] + original_sep)
                target_orders = targets
                perms = target_to_current_perms(meta, target_orders)
            elif reference_order_store is None:
                perms = {li: torch.argsort(scores[li], descending=True,
                                           stable=True).tolist()
                         for li in range(meta["num_layers"])}

        if args.order == "visionzip":
            method = "visionzip_image_only"
        elif args.order == "importance" and args.separator_tail:
            method = "calib_importance_sep_tail"
        else:
            method = args.order
        provenance = None
        if args.record_layout_provenance or args.order in ("visionzip", "raster"):
            history = list(meta.get("transformation_history", []))
            history.append({
                "operation": f"posthoc_target_{method}",
                "source_meta_sha256": before_meta_sha,
                "reference_order_meta_sha256": (
                    _file_sha256(reference_order_store / str(e["image_id"])
                                 / "meta.json")
                    if reference_order_store is not None else None),
            })
            global_shared = not (target_orders and
                                 isinstance(target_orders[0], list))
            provenance = {
                "physical_layout": method,
                "layout_method": method,
                "layout_source": ("posthoc_fresh_raster" if
                                  args.order == "visionzip" else
                                  "posthoc_target_order"),
                "layout_uses_dataset_question": args.order == "importance",
                "llm_used_for_layout_scoring": args.order == "importance",
                "calibration_questions": (args.calib_questions
                                          if args.order == "importance" else 0),
                "calibration_order_reused_from": (
                    str(reference_order_store)
                    if reference_order_store is not None else None),
                "global_order_all_layers": bool(global_shared),
                "separator_tail": bool(
                    args.order in ("visionzip", "morton")
                    or args.separator_tail),
                "separator_policy": ("stable_tail_plus_sidecar" if
                                     (args.order in ("visionzip", "morton")
                                      or args.separator_tail) else
                                     "original_positions_plus_sidecar"),
                "composed_from_store": args.order != "visionzip",
                "composed_from_reordered_layout": args.order != "visionzip",
                "transformation_history": history,
            }
            if target_orders is not None:
                if global_shared:
                    provenance["permutation_sha256"] = permutation_sha256(
                        target_orders)
                    provenance["inverse_permutation_sha256"] = \
                        permutation_sha256(mapping_from_perm(target_orders))
                else:
                    provenance["permutation_sha256_per_layer"] = [
                        permutation_sha256(row) for row in target_orders]

        rewrite_t0 = time.perf_counter()
        shared = commit(d, meta, perms, metadata_update=provenance,
                        identity_layout=args.order == "raster")
        rewrite_ms = (time.perf_counter() - rewrite_t0) * 1e3
        meta = load_meta(d)
        sep_ms = 0.0
        if (args.rebuild_separator_sidecar or args.order == "visionzip"
                or args.separator_tail):
            sep_t0 = time.perf_counter()
            write_separator_sidecar_from_store(d, meta)
            sep_ms = (time.perf_counter() - sep_t0) * 1e3
            meta = load_meta(d)

        layout_bytes = 0
        if args.order == "visionzip":
            artifact = {
                "schema_version": 1,
                "image_id": str(e["image_id"]),
                "importance_source":
                    "vision_encoder_penultimate_cls_to_patch_attention_head_sum",
                "token_score_original": token_score,
                "stored_to_original": torch.tensor(target_orders,
                                                    dtype=torch.int32),
                "original_to_stored": torch.tensor(
                    mapping_from_perm(target_orders), dtype=torch.int32),
                "newline_original": torch.tensor(meta["newline_idx"],
                                                 dtype=torch.int32),
                "layout_uses_dataset_question": False,
                "llm_used_for_layout_scoring": False,
                "calibration_questions": 0,
                "prototype_posthoc": True,
            }
            torch.save(artifact, d / "visionzip_layout.pt")
            layout_bytes = (d / "visionzip_layout.pt").stat().st_size

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        profiles.append({
            "image_id": str(e["image_id"]),
            "method": method,
            "source_meta_sha256": before_meta_sha,
            "final_meta_sha256": _file_sha256(d / "meta.json"),
            "vision_saliency_ms": saliency_ms,
            "kv_repack_ms": rewrite_ms,
            "separator_sidecar_ms": sep_ms,
            "total_reorder_ms": elapsed_ms,
            "mapping_metadata_bytes": ((d / "meta.json").stat().st_size
                                       + layout_bytes),
            "shared_order": bool(shared),
            "separator_tail": bool(meta.get("separator_tail", False)),
            "layout_uses_dataset_question": args.order == "importance",
            "llm_used_for_layout_scoring": args.order == "importance",
            "calibration_questions": (args.calib_questions
                                      if args.order == "importance" else 0),
            "calibration_order_reused_from": (
                str(reference_order_store)
                if reference_order_store is not None else None),
        })
        print(f"[{i+1}/{len(index)}] {e['image_id']}: {args.order} order "
              f"({'shared' if shared else 'per-layer'}) "
              f"({elapsed_ms/1e3:.1f}s)")
    print("reorder done")
    if args.profile_out:
        out = Path(args.profile_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema_version": 1,
            "command": shlex.join([sys.executable, *sys.argv]),
            "store": str(store.resolve()),
            "order": args.order,
            "separator_tail": bool(args.separator_tail),
            "layout_uses_dataset_question": args.order == "importance",
            "llm_used_for_layout_scoring": args.order == "importance",
            "calibration_questions": (args.calib_questions
                                      if args.order == "importance" else 0),
            "reference_order_store": (
                str(reference_order_store)
                if reference_order_store is not None else None),
            "n_images": len(profiles),
            "rows": profiles,
        }, indent=1))
        print(f"profile: {out}")


if __name__ == "__main__":
    main()

"""Analysis-only importance concentration for physical Visual-KV layouts.

VisionZip image saliency is read from the direct image-only store in ORIGINAL
visual-token coordinates.  SparseVLM calibration importance is recomputed once
from the protected calib=4 store, mapped back to ORIGINAL coordinates, and is
used only by this offline analysis.  It is never passed to a writer, selector,
or online request.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import CHUNK_SIZE, MODEL_ID, STORE_DIR
from mmimpress.cvpr25 import budget_chunk_count, permutation_sha256
from mmimpress.dataset import load_index
from mmimpress.model import LlavaRunner
from mmimpress.reorder import llava_visual_order
from mmimpress.serve import BIAS, ImageContext, Server, calibrate_image
from mmimpress.store import load_meta


INDEX_SHA256 = "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
EVAL_WORKLOAD_SHA256 = "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
CALIB_WORKLOAD_SHA256 = "992cc89a81a6cadf363b65b58f8f89fabfc0a1a17358559069f8c0b6cabc1e71"
CANONICAL_STORE_CONTENT_SHA256 = "e570a6847743a203fc1e2892d736ebe8aa946647cbb0e280f212388da2c09d68"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def workload_sha(index, start, count):
    blob = "\n".join(
        f"{entry['image_id']}\t{q['question_id']}"
        for entry in index for q in entry["questions"][start:start + count]
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def layer_orders(meta):
    raw = meta.get("order")
    L, vn = int(meta["num_layers"]), int(meta["v_token_num"])
    if not raw:
        return [list(range(vn)) for _ in range(L)]
    return raw if meta.get("order_is_per_layer") else [raw for _ in range(L)]


def stored_scores_to_original(scores, orders):
    """scores[L, stored] -> scores[L, original]."""
    answer = torch.empty_like(scores, dtype=torch.float64)
    for li, order in enumerate(orders):
        idx = torch.tensor(order, dtype=torch.long)
        answer[li, idx] = scores[li].double()
    return answer


def validate_layout_order(name, orders, meta, expect_tail=False):
    vn = int(meta["v_token_num"])
    expected = list(range(vn))
    sep = set(int(x) for x in meta["newline_idx"])
    for li, order in enumerate(orders):
        if len(order) != vn or sorted(int(x) for x in order) != expected:
            raise AssertionError(f"invalid {name} permutation layer {li}")
        if expect_tail:
            tail = [int(x) for x in order[-len(sep):]] if sep else []
            if tail != sorted(sep):
                raise AssertionError(f"{name} separators are not stable-tail")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--calib4-store", default=str(STORE_DIR))
    ap.add_argument("--visionzip-store", required=True)
    ap.add_argument("--matched-calib-store", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fractions", default="0.10,0.20,0.25,0.30,0.50")
    ap.add_argument("--calib-questions", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_store = Path(args.calib4_store).resolve()
    vision_store = Path(args.visionzip_store).resolve()
    matched_store = (Path(args.matched_calib_store).resolve()
                     if args.matched_calib_store else None)
    if calib_store != STORE_DIR.resolve():
        raise AssertionError("calibration analysis must use canonical calib4 store")
    if vision_store == calib_store or (matched_store and matched_store == calib_store):
        raise AssertionError("new output stores must be disjoint from calib4")

    index_path = Path(args.index)
    index = load_index(index_path)
    if len(index) != 40 or sha256(index_path) != INDEX_SHA256:
        raise AssertionError("index is not frozen GQA-40")
    if workload_sha(index, 4, 6) != EVAL_WORKLOAD_SHA256:
        raise AssertionError("evaluation workload differs from questions[4:10]")
    if workload_sha(index, 0, args.calib_questions) != CALIB_WORKLOAD_SHA256:
        raise AssertionError("analysis-only calibration workload differs")
    fractions = [float(x) for x in args.fractions.split(",")]
    if fractions != [0.10, 0.20, 0.25, 0.30, 0.50]:
        raise AssertionError("required cumulative fractions changed")

    runner = LlavaRunner().load()
    server = Server(runner)
    rows = []
    raw_scores = {}
    query_count = 0
    start = time.time()
    for image_i, entry in enumerate(index):
        image_id = str(entry["image_id"])
        calib_dir = calib_store / image_id
        vision_dir = vision_store / image_id
        cm = load_meta(calib_dir)
        vm = load_meta(vision_dir)
        for key in ("v_token_num", "num_layers", "num_heads", "head_dim",
                    "chunk_size", "newline_idx", "prefix_input_ids"):
            if cm[key] != vm[key]:
                raise AssertionError(f"store geometry mismatch {image_id}/{key}")
        if vm.get("physical_layout") != "visionzip_image_only":
            raise AssertionError(f"wrong VisionZip provenance: {image_id}")
        if (vm.get("layout_uses_dataset_question") is not False
                or vm.get("llm_used_for_layout_scoring") is not False
                or int(vm.get("calibration_questions", -1)) != 0):
            raise AssertionError(f"query-dependent VisionZip meta: {image_id}")

        artifact = torch.load(vision_dir / "visionzip_layout.pt",
                              weights_only=True)
        vision_score = artifact["token_score_original"].double()
        vn, L = int(cm["v_token_num"]), int(cm["num_layers"])
        if vision_score.shape != (vn,):
            raise AssertionError(f"bad VisionZip token score: {image_id}")
        sep = sorted(int(x) for x in cm["newline_idx"])
        finite = torch.isfinite(vision_score)
        if torch.nonzero(~finite).flatten().tolist() != sep:
            raise AssertionError(f"VisionZip separator mask mismatch: {image_id}")

        ctx = ImageContext(calib_dir, runner.model.device, drop_cache=False)
        questions = [q["question"]
                     for q in entry["questions"][:args.calib_questions]]
        calibration_stored = calibrate_image(server, ctx, questions)
        query_count += len(questions)
        ctx.close()
        BIAS.clear()
        calibration_original = stored_scores_to_original(
            calibration_stored, layer_orders(cm))
        raw_scores[image_id] = calibration_original.float()

        base, (hi_h, hi_w) = int(cm["base_grid"]), cm["hires_grid"]
        morton, _ = llava_visual_order(base, int(hi_h), int(hi_w), sep, vn)
        layouts = {
            "raster": [list(range(vn)) for _ in range(L)],
            "morton": [morton for _ in range(L)],
            "visionzip_image_only": layer_orders(vm),
            "calib4_importance_legacy": layer_orders(cm),
        }
        if matched_store:
            mm = load_meta(matched_store / image_id)
            for key in ("v_token_num", "num_layers", "newline_idx"):
                if mm[key] != cm[key]:
                    raise AssertionError(
                        f"matched calibration geometry mismatch {image_id}/{key}")
            if mm.get("physical_layout") != "calib_importance_sep_tail":
                raise AssertionError(f"wrong matched layout meta: {image_id}")
            matched_orders = layer_orders(mm)
            legacy_orders = layouts["calib4_importance_legacy"]
            sep_set = set(sep)
            exact_matched_targets = [
                [int(x) for x in order if int(x) not in sep_set] + sep
                for order in legacy_orders
            ]
            if matched_orders != exact_matched_targets:
                raise AssertionError(
                    f"matched calib4 does not preserve legacy normal order: "
                    f"{image_id}")
            layouts["calib4_importance_sep_tail"] = matched_orders

        validate_layout_order("raster", layouts["raster"], cm)
        validate_layout_order("morton", layouts["morton"], cm, True)
        validate_layout_order(
            "visionzip_image_only", layouts["visionzip_image_only"], cm, True)
        validate_layout_order(
            "calib4_importance_legacy",
            layouts["calib4_importance_legacy"], cm)
        if "calib4_importance_sep_tail" in layouts:
            validate_layout_order(
                "calib4_importance_sep_tail",
                layouts["calib4_importance_sep_tail"], cm, True)

        sources = {
            "visionzip_image_saliency": vision_score.unsqueeze(0).repeat(L, 1),
            "sparsevlm_calib4_analysis_only": calibration_original,
        }
        normal_mask = finite
        for layout, orders in layouts.items():
            for li, order in enumerate(orders):
                order_t = torch.tensor(order, dtype=torch.long)
                for fraction in fractions:
                    nc = int(cm["n_chunks_per_layer"])
                    k = budget_chunk_count(nc, fraction)
                    end = min(k * int(cm["chunk_size"]), vn)
                    selected_original = order_t[:end]
                    selected_normal = selected_original[
                        normal_mask[selected_original]]
                    for source, score_matrix in sources.items():
                        score = score_matrix[li]
                        total_mass = float(score[normal_mask].sum())
                        selected_mass = float(score[selected_normal].sum())
                        coverage = selected_mass / total_mass if total_mass else math.nan
                        rows.append({
                            "image_id": image_id,
                            "layer": li,
                            "layout": layout,
                            "importance_source": source,
                            "analysis_only_calibration":
                                source == "sparsevlm_calib4_analysis_only",
                            "physical_fraction_requested": fraction,
                            "selected_chunk_count": k,
                            "total_chunk_count": nc,
                            "selected_physical_rows": end,
                            "selected_normal_tokens": int(selected_normal.numel()),
                            "total_normal_tokens": int(normal_mask.sum()),
                            "realized_normal_token_fraction": (
                                int(selected_normal.numel()) /
                                int(normal_mask.sum())),
                            "selected_importance_mass": selected_mass,
                            "total_importance_mass": total_mass,
                            "importance_mass_coverage": coverage,
                            "separator_tokens": len(sep),
                            "separator_tail": layout in {
                                "morton", "visionzip_image_only",
                                "calib4_importance_sep_tail"},
                            "prefix_rule": "chunks range(0,k)",
                            "order_sha256": permutation_sha256(order),
                        })
        torch.cuda.empty_cache()
        print(f"[{image_i + 1}/40] {image_id}: coverage/calibration done "
              f"({time.time() - start:.0f}s)")

    fields = list(rows[0])
    coverage_path = out_dir / "importance_coverage.csv"
    with coverage_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    keys = sorted({(r["layout"], r["importance_source"],
                    r["physical_fraction_requested"]) for r in rows})
    for layout, source, fraction in keys:
        group = [r for r in rows if r["layout"] == layout
                 and r["importance_source"] == source
                 and r["physical_fraction_requested"] == fraction]
        selected = sum(float(r["selected_importance_mass"]) for r in group)
        total = sum(float(r["total_importance_mass"]) for r in group)
        summary.append({
            "layout": layout,
            "importance_source": source,
            "physical_fraction_requested": fraction,
            "n_image_layers": len(group),
            "macro_mean_coverage": float(np.mean([
                float(r["importance_mass_coverage"]) for r in group])),
            "global_mass_weighted_coverage": selected / total,
            "mean_realized_normal_token_fraction": float(np.mean([
                float(r["realized_normal_token_fraction"]) for r in group])),
        })
    (out_dir / "coverage_summary.json").write_text(json.dumps({
        "schema_version": 1,
        "analysis_only": True,
        "coverage_used_for_layout_or_serving": False,
        "sparsevlm_scores_used_for_layout": False,
        "visionzip_layout_uses_dataset_question": False,
        "llm_used_for_visionzip_layout_scoring": False,
        "visionzip_calibration_questions": 0,
        "matched_calib_exact_legacy_normal_order_plus_separator_tail":
            matched_store is not None,
        "calibration_importance_source":
            "fresh calibrate_image over questions[0:4], analysis only",
        "n_images": 40,
        "n_calibration_questions_processed": query_count,
        "n_rows": len(rows),
        "fractions": fractions,
        "summary": summary,
        "inputs": {
            "index": str(index_path.resolve()),
            "index_sha256": INDEX_SHA256,
            "evaluation_workload_sha256": EVAL_WORKLOAD_SHA256,
            "calibration_workload_sha256": CALIB_WORKLOAD_SHA256,
            "calib4_store": str(calib_store),
            "calib4_store_expected_content_sha256":
                CANONICAL_STORE_CONTENT_SHA256,
            "visionzip_store": str(vision_store),
            "matched_calib_store": str(matched_store) if matched_store else None,
            "model": MODEL_ID,
            "chunk_size": CHUNK_SIZE,
        },
        "command": shlex.join([sys.executable, *sys.argv]),
        "coverage_csv_sha256": sha256(coverage_path),
    }, indent=1))
    torch.save({
        "schema_version": 1,
        "analysis_only": True,
        "used_for_layout_or_serving": False,
        "scores_original_by_image": raw_scores,
    }, out_dir / "calibration_scores_analysis_only.pt")
    print(f"wrote {coverage_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

"""Quota-matched two-dimensional sampling of original LLaVA-NeXT visual KV.

This module changes only which non-dominant *original IDs* fill the auxiliary
budget.  It consumes the frozen IndexUniform plan for its dominant set, exact
25% budget, saliency ranking, and base/high-resolution auxiliary quotas.  No
feature or K/V value is combined or modified.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Sequence

import torch

from mmimpress.contextual_kv25 import uniform_target_ids


LAYOUT_POLICY = "visionzip_spatial_original_v1"
BRANCHES = ("base", "high")


def anyres_token_coordinates(runner: Any, image_size: Sequence[int] | torch.Tensor,
                             v_num: int, base_side: int | None = None
                             ) -> dict[str, Any]:
    """Map each packed visual ID to its actual branch patch-grid coordinate.

    The source patch labels follow installed Transformers' LLaVA-NeXT
    ``pack_image_features`` path: ``(tile_y,tile_x,patch_y,patch_x)`` is
    permuted to a tiled patch grid, unpadded, and interleaved with row newline
    tokens.  Coordinates describe this *representation grid*, not original
    image pixel coordinates.  Newlines have ``None`` records.
    """
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape, unpad_image)

    if torch.is_tensor(image_size):
        size = [int(x) for x in image_size.detach().cpu().reshape(-1)]
    else:
        size = [int(x) for x in image_size]
    if len(size) != 2 or min(size) < 1:
        raise ValueError("image_size must contain positive height and width")
    if isinstance(v_num, bool) or not isinstance(v_num, Integral) or v_num < 1:
        raise ValueError("v_num must be a positive integer")
    cfg = runner.cfg
    vc = cfg.vision_config
    side = (int(base_side) if base_side is not None
            else int(vc.image_size // vc.patch_size))
    if side < 1:
        raise ValueError("base_side must be positive")
    nph, npw = get_anyres_image_grid_shape(
        size, cfg.image_grid_pinpoints, vc.image_size)
    nph, npw = int(nph), int(npw)
    if nph < 1 or npw < 1:
        raise ValueError("invalid AnyRes tile grid")

    labels = torch.arange(nph * npw * side * side, dtype=torch.int64)
    tiled = labels.view(nph, npw, side, side).permute(0, 2, 1, 3)
    grid = tiled.reshape(nph * side, npw * side)
    kept = unpad_image(grid.unsqueeze(0), size)[0]
    hi_h, hi_w = map(int, kept.shape)
    base_count = side * side
    if base_count + hi_h * (hi_w + 1) != v_num:
        raise ValueError("AnyRes coordinate geometry disagrees with visual span")
    if len(set(kept.reshape(-1).tolist())) != hi_h * hi_w:
        raise AssertionError("unpadding repeated a high-resolution patch")

    records: list[dict[str, Any] | None] = [None] * v_num
    for row in range(side):
        for col in range(side):
            original = row * side + col
            records[original] = {
                "original_visual_id": original, "branch": "base",
                "row": row, "column": col,
                "grid_height": side, "grid_width": side,
                "x": (col + .5) / side, "y": (row + .5) / side,
                "source_subimage_index": 0,
                "source_tile_row": None, "source_tile_column": None,
                "source_patch_row": row, "source_patch_column": col,
            }
    newline_ids = []
    for row in range(hi_h):
        for col in range(hi_w):
            original = base_count + row * (hi_w + 1) + col
            label = int(kept[row, col])
            tile_id, patch_id = divmod(label, base_count)
            tile_row, tile_col = divmod(tile_id, npw)
            patch_row, patch_col = divmod(patch_id, side)
            records[original] = {
                "original_visual_id": original, "branch": "high",
                "row": row, "column": col,
                "grid_height": hi_h, "grid_width": hi_w,
                "x": (col + .5) / hi_w, "y": (row + .5) / hi_h,
                "source_subimage_index": tile_id + 1,
                "source_tile_row": tile_row, "source_tile_column": tile_col,
                "source_patch_row": patch_row,
                "source_patch_column": patch_col,
            }
        newline_ids.append(base_count + row * (hi_w + 1) + hi_w)
    if any(records[i] is not None for i in newline_ids):
        raise AssertionError("a structural newline received a patch coordinate")
    if sum(record is not None for record in records) != base_count + hi_h * hi_w:
        raise AssertionError("coordinate mapping omitted a content token")
    return {
        "coordinate_space": "model_input_patch_grid",
        "image_size_height_width": size,
        "base_grid": {"height": side, "width": side},
        "high_grid": {"height": hi_h, "width": hi_w},
        "tile_grid": {"height": nph, "width": npw},
        "structural_original_ids": newline_ids,
        "records": records,
    }


def make_spatial_regions(height: int, width: int, q: int
                         ) -> list[dict[str, Any]]:
    """Partition a branch into exactly ``q`` deterministic equal-area cells.

    Coordinates are normalized to [0, 1].  Every interval is half-open except
    its final upper edge, which includes 1.  Patch centers are strictly inside
    [0, 1], so exact integer arithmetic can assign all real patch positions.
    """
    if any(isinstance(x, bool) or not isinstance(x, Integral)
           for x in (height, width, q)) or height < 1 or width < 1 \
            or q < 0 or q > height * width:
        raise ValueError("invalid spatial grid or region count")
    if q == 0:
        return []
    r_min = (q + width - 1) // width
    r_max = min(q, height)
    r_ideal = math.floor(math.sqrt(q * height / width) + .5)
    rows = min(max(r_ideal, r_min), r_max)
    regions: list[dict[str, Any]] = []
    start = 0
    for row_bin in range(rows):
        cols = q // rows + (1 if row_bin < q % rows else 0)
        for col_bin in range(cols):
            regions.append({
                "region_id": len(regions),
                "row_bin": row_bin, "column_bin": col_bin,
                "rows": rows, "columns_in_row": cols,
                "row_start_unit": start,
                "x_min": col_bin / cols,
                "x_max": (col_bin + 1) / cols,
                "y_min": start / q,
                "y_max": (start + cols) / q,
                "x_center": (col_bin + .5) / cols,
                "y_center": (start + cols / 2) / q,
                "normalized_area": 1 / q,
                "boundary_rule": "half_open_except_final_upper_edge",
            })
        start += cols
    assert len(regions) == q and start == q
    return regions


def _region_for_patch(record: dict[str, Any], regions: list[dict[str, Any]],
                      height: int, width: int, q: int) -> int:
    row, col = int(record["row"]), int(record["column"])
    # floor(q * (row+.5)/H) assigns a boundary to the following row band.
    y_unit = ((2 * row + 1) * q) // (2 * height)
    for region in regions:
        start = int(region["row_start_unit"])
        cols = int(region["columns_in_row"])
        if start <= y_unit < start + cols:
            x_bin = ((2 * col + 1) * cols) // (2 * width)
            if x_bin == region["column_bin"]:
                return int(region["region_id"])
    raise AssertionError("patch center did not belong to a spatial region")


def _distance_key(record: dict[str, Any], region: dict[str, Any],
                  height: int, width: int, q: int) -> tuple[int, int]:
    """Exact integer comparison of the required scaled squared distance.

    The common denominator within one region is ``4 * cols² * q²``.  Using
    integers keeps mathematically tied patch centers tied across platforms.
    """
    cols = int(region["columns_in_row"])
    col_bin = int(region["column_bin"])
    start = int(region["row_start_unit"])
    dx_num = ((2 * int(record["column"]) + 1) * cols
              - width * (2 * col_bin + 1))
    dy_num = ((2 * int(record["row"]) + 1) * q
              - height * (2 * start + cols))
    scaled_sq = dx_num * dx_num * q * q + dy_num * dy_num * cols * cols
    return scaled_sq, int(record["original_visual_id"])


def select_spatial_branch(records: Sequence[dict[str, Any] | None],
                          remaining_ids: Sequence[int], q: int,
                          height: int, width: int) -> dict[str, Any]:
    """Choose one original token per nonempty region, then fill empty cells.

    The first pass visits every nonempty region before the second pass visits
    empty regions in row-major order.  Both nearest-token decisions break exact
    distance ties with the lower original visual ID.
    """
    ids = [int(i) for i in remaining_ids]
    if ids != sorted(set(ids)):
        raise ValueError("branch remaining IDs must be strictly increasing")
    if isinstance(q, bool) or not isinstance(q, Integral) or q < 0 \
            or q > len(ids):
        raise ValueError("branch quota exceeds available non-dominant tokens")
    regions = make_spatial_regions(height, width, q)
    for original in ids:
        if original < 0 or original >= len(records):
            raise ValueError("branch ID outside coordinate records")
        record = records[original]
        if record is None or record["original_visual_id"] != original \
                or record["grid_height"] != height \
                or record["grid_width"] != width \
                or not 0 <= record["row"] < height \
                or not 0 <= record["column"] < width:
            raise ValueError("branch ID has invalid patch-grid coordinate")
    candidates: list[list[int]] = [[] for _ in regions]
    for original in ids:
        if q:
            region_id = _region_for_patch(records[original], regions,
                                          height, width, q)
            candidates[region_id].append(original)

    selected: set[int] = set()
    selected_by_region: dict[int, tuple[int, bool]] = {}
    for region_id, region in enumerate(regions):
        if candidates[region_id]:
            chosen = min(candidates[region_id],
                         key=lambda i: _distance_key(records[i], region,
                                                     height, width, q))
            if chosen in selected:
                raise AssertionError("nonempty regions selected the same token")
            selected.add(chosen)
            selected_by_region[region_id] = (chosen, False)
    empty_count = sum(not candidate for candidate in candidates)
    for region_id, region in enumerate(regions):
        if candidates[region_id]:
            continue
        available = [i for i in ids if i not in selected]
        if not available:
            raise AssertionError("empty-region fallback ran out of tokens")
        chosen = min(available, key=lambda i: _distance_key(
            records[i], region, height, width, q))
        selected.add(chosen)
        selected_by_region[region_id] = (chosen, True)
    if len(selected) != q or len(selected_by_region) != q:
        raise AssertionError("spatial branch did not satisfy its exact quota")

    for region_id, region in enumerate(regions):
        chosen, fallback = selected_by_region[region_id]
        scaled_sq, _ = _distance_key(records[chosen], region,
                                     height, width, q)
        cols = int(region["columns_in_row"])
        region["original_candidate_count"] = len(candidates[region_id])
        region["originally_empty"] = not bool(candidates[region_id])
        region["fallback"] = fallback
        region["selected_original_id"] = chosen
        region["selected_coordinate"] = {
            "x": records[chosen]["x"], "y": records[chosen]["y"]}
        region["distance_to_center"] = math.sqrt(scaled_sq) / (2 * cols * q)
    return {
        "grid_height": height, "grid_width": width,
        "quota": q, "remaining_count": len(ids),
        "region_rows": regions[0]["rows"] if regions else 0,
        "regions": regions,
        "selected_original_ids": [selected_by_region[i][0]
                                  for i in range(len(regions))],
        "empty_region_count": empty_count,
        "fallback_count": sum(region["fallback"] for region in regions),
    }


def build_spatial_selection_plan(index_plan: dict[str, Any],
                                 coordinates: dict[str, Any]
                                 ) -> dict[str, Any]:
    """Build a full spatial SSD permutation from the frozen IndexUniform plan.

    ``index_plan`` must be the existing ``variant='uniform'`` plan for this
    image and auxiliary budget.  Its selected IDs determine branch quotas,
    while its dominant IDs and saliency ranking are copied exactly.
    """
    if index_plan.get("variant") != "uniform":
        raise ValueError("spatial selection requires an IndexUniform plan")
    records = coordinates.get("records")
    order = [int(x) for x in index_plan["stored_to_original"]]
    if not isinstance(records, list) or len(records) != len(order):
        raise ValueError("coordinate records disagree with visual span")
    separators = [int(x) for x in index_plan["structural_original_ids"]]
    if separators != [int(x) for x in coordinates["structural_original_ids"]] \
            or any(records[i] is not None for i in separators):
        raise ValueError("structural coordinates disagree with IndexUniform")
    content_rank = [int(x) for x in index_plan["content_saliency_ranking"]]
    dominant = [int(x) for x in index_plan["dominant_ids"]]
    dominant_set = set(dominant)
    remaining = [int(x) for x in index_plan["remaining_ids"]]
    uniform_aux = [int(x) for x in index_plan["contextual_ids"]]
    k = int(index_plan["k"])
    q = int(index_plan["k_context"])
    if (len(content_rank) != int(index_plan["n_content"])
            or len(dominant) != int(index_plan["k_dominant"])
            or len(set(content_rank)) != len(content_rank)
            or sorted(remaining) != remaining
            or set(remaining) != set(content_rank) - dominant_set
            or uniform_aux != uniform_target_ids(remaining, q)
            or len(set(uniform_aux)) != q or len(dominant) + q != k):
        raise ValueError("IndexUniform plan is internally inconsistent")
    if set(content_rank) != {i for i, record in enumerate(records)
                             if record is not None}:
        raise ValueError("coordinate content IDs disagree with IndexUniform")
    for original, record in enumerate(records):
        if record is None:
            continue
        branch = record.get("branch")
        grid = coordinates["base_grid" if branch == "base" else "high_grid"] \
            if branch in BRANCHES else None
        if grid is None or record.get("original_visual_id") != original \
                or record.get("grid_height") != grid["height"] \
                or record.get("grid_width") != grid["width"]:
            raise ValueError("invalid coordinate record")

    quotas = {branch: sum(records[i]["branch"] == branch for i in uniform_aux)
              for branch in BRANCHES}
    branch_results = {}
    spatial_aux = []
    for branch in BRANCHES:
        branch_remaining = [i for i in remaining
                            if records[i]["branch"] == branch]
        grid = coordinates["base_grid" if branch == "base" else "high_grid"]
        result = select_spatial_branch(records, branch_remaining,
                                       quotas[branch], int(grid["height"]),
                                       int(grid["width"]))
        branch_results[branch] = result
        spatial_aux.extend(result["selected_original_ids"])
    selected_set = dominant_set | set(spatial_aux)
    if (len(spatial_aux) != q or len(set(spatial_aux)) != q
            or len(selected_set) != k or dominant_set.intersection(spatial_aux)):
        raise AssertionError("spatial selection violated the fixed budget")
    selected_order = [i for i in content_rank if i in selected_set]
    unselected_order = [i for i in content_rank if i not in selected_set]
    stored_to_original = selected_order + unselected_order + separators
    if len(stored_to_original) != len(order) \
            or sorted(stored_to_original) != list(range(len(order))):
        raise AssertionError("spatial physical order is not a permutation")
    original_to_stored = [0] * len(order)
    for stored, original in enumerate(stored_to_original):
        original_to_stored[original] = stored
    if q == 0 and stored_to_original != order:
        raise AssertionError("zero auxiliary quota changed the old order")

    union = set(uniform_aux) | set(spatial_aux)
    plan = dict(index_plan)
    plan.update({
        "layout_policy": LAYOUT_POLICY,
        "selection_source": "turn1_image_only_saliency_and_geometry",
        "variant": "spatial_uniform",
        "contextual_ids": spatial_aux,
        "target_ids": [],
        "assignments": {}, "representative_ids": [],
        "cluster_sizes": [], "cluster_size_by_target": {},
        "cluster_members": {},
        "selected_original_ids": selected_order,
        "stored_to_original": stored_to_original,
        "original_to_stored": original_to_stored,
        "spatial": {
            "definition": "base_high_quota_matched_2d_stratified_v1",
            "coordinates": coordinates,
            "index_uniform_aux_ids": uniform_aux,
            "branch_quotas": quotas,
            "branch_results": branch_results,
            "selected_aux_ids": spatial_aux,
            "jaccard_index_vs_spatial": (
                len(set(uniform_aux) & set(spatial_aux)) / len(union)
                if union else 1.0),
            "added_ids": sorted(set(spatial_aux) - set(uniform_aux)),
            "removed_ids": sorted(set(uniform_aux) - set(spatial_aux)),
        },
    })
    return plan

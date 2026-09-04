"""KV reordering for image prefixes (IMPRESS 4.4.1, adapted to a 2-D prefix).

IMPRESS packs important KVs into dense chunks so that fetching them touches few
chunks.  For a text prefix the token axis is 1-D and importance is fairly
clustered, so sorting by average importance is enough.  An image prefix is a 2-D
patch grid flattened in raster order, and that makes the default layout
pathological: with a 24x24 grid a 64-token chunk is a ~2.7-row horizontal STRIP,
while an object is a compact 2-D BLOB, so one object crosses many strips.  Every
chunk ends up holding at least one selected patch and chunk skipping buys
nothing.

Three orders are provided so the effect can be measured rather than assumed:

  identity    raster order, i.e. what the model emits (baseline)
  importance  IMPRESS 4.4.1: descending average importance, per (layer, head)
              since SparseVLM scores are per head
  morton      Z-order over the patch grid, so a chunk is a ~8x8 spatial TILE
              instead of a strip; query-independent, needs no statistics

Mapping convention is IMPRESS's: stored = original[perm] and
stored[mapping] = original, i.e. mapping[orig_pos] = stored_pos.  Both
directions are needed -- selection produces original indices and the loader
needs stored positions.
"""
from __future__ import annotations

import torch


def mapping_from_perm(perm):
    """perm (stored = original[perm]) -> mapping (mapping[orig] = stored)."""
    mapping = [0] * len(perm)
    for stored_pos, orig_pos in enumerate(perm):
        mapping[orig_pos] = stored_pos
    return mapping


def importance_order(importance):
    """Stable descending-importance permutation for one head."""
    imp = [float(x) for x in importance]
    perm = sorted(range(len(imp)), key=lambda i: (-imp[i], i))
    return perm, mapping_from_perm(perm)


# ------------------------------------------------------------------ morton
def _morton_key(x: int, y: int) -> int:
    key = 0
    for b in range(16):
        key |= ((x >> b) & 1) << (2 * b)
        key |= ((y >> b) & 1) << (2 * b + 1)
    return key


def morton_grid_order(h: int, w: int):
    """Z-order permutation of a raster-flattened h x w grid."""
    cells = [(r * w + c, _morton_key(c, r)) for r in range(h) for c in range(w)]
    cells.sort(key=lambda t: t[1])
    return [i for i, _ in cells]


def llava_visual_order(base: int, hi_h: int, hi_w: int, newline_idx,
                       v_num: int):
    """Morton order for LLaVA-NeXT's visual block.

    The block is [base*base low-res patches | hi_h*(hi_w+1) high-res patches],
    where the last column of every high-res row is a learned row separator.
    Each sub-grid is Z-ordered independently; separators keep their relative
    order and are appended at the end, since they are always retained and are
    better off not fragmenting the spatial tiles.
    """
    nl = set(int(i) for i in newline_idx)
    perm = [i for i in morton_grid_order(base, base)]           # low-res
    hi_start = base * base
    hi_cells = [(r, c) for r in range(hi_h) for c in range(hi_w)]
    hi_cells.sort(key=lambda rc: _morton_key(rc[1], rc[0]))
    for r, c in hi_cells:
        idx = hi_start + r * (hi_w + 1) + c
        if idx not in nl:
            perm.append(idx)
    perm.extend(sorted(nl))
    assert len(perm) == v_num, (len(perm), v_num)
    assert len(set(perm)) == v_num, "morton order is not a permutation"
    return perm, mapping_from_perm(perm)


# ------------------------------------------------------------- diagnostics
def chunks_touched(stored_positions, chunk_size: int) -> int:
    return len({int(p) // chunk_size for p in stored_positions})


def chunk_fraction(selected_original, mapping, chunk_size: int, v_num: int):
    """Fraction of a head's chunks that a selection forces to be read.

    mapping=None means identity (raster).  This is the number P4c measured as
    0.993 on the token-major raster store; the whole point of reordering is to
    drive it toward k/chunk_size/v_num.
    """
    stored = ([mapping[int(i)] for i in selected_original]
              if mapping is not None else [int(i) for i in selected_original])
    n_total = (v_num + chunk_size - 1) // chunk_size
    return chunks_touched(stored, chunk_size) / n_total


def apply_perm(tensor: torch.Tensor, perm, dim: int = 0):
    """stored = original[perm] along `dim`."""
    idx = torch.as_tensor(perm, dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim, idx)


def restore(tensor: torch.Tensor, mapping, dim: int = 0):
    """original = stored[mapping] along `dim` (IMPRESS's s'[m])."""
    idx = torch.as_tensor(mapping, dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim, idx)

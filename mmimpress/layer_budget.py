"""Layer-wise visual-KV row budget for PrefixStore-KV25 (``layerwise_v1``).

Pure, CPU-testable arithmetic.  Nothing in this module reads a question, a
score, a store or the model.  The only inputs are a fixed per-model ratio
vector ``r`` (one entry per decoder layer) and an image's original visual
content count ``N``.

Definitions (frozen in ``docs/layer_budget_kv25_contract.md``):

* total row budget per image ``B(N) = L * ceil(0.25 * N)`` -- identical to the
  uniform Ours-KV25 store (``L`` layers, ``ceil(N/4)`` rows each);
* ``sum(r) = L * 0.25`` and ``r_min <= r_l <= r_max`` for every layer;
* ``k_l`` = Hamilton largest-remainder apportionment of ``B(N)`` with quotas
  ``r_l * N``: ``floor`` first, then one extra row per layer in descending
  fractional-remainder order (ties: lower layer index first), repeated until
  the total is exactly ``B(N)``; ``k_l <= N`` always;
* the uniform vector (all ``0.25``) reproduces ``k_l = ceil(N/4)`` exactly.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real

NUM_LAYERS = 32
TOTAL_RATIO = 0.25
R_MIN = 0.10
R_MAX = 0.60
POLICY = "layerwise_v1"
SUM_TOL = 1e-9


def _is_real(x) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)


def uniform_ratios(num_layers: int = NUM_LAYERS,
                   ratio: float = TOTAL_RATIO) -> list[float]:
    return [float(ratio)] * int(num_layers)


def validate_ratios(ratios, num_layers: int = NUM_LAYERS,
                    total_ratio: float = TOTAL_RATIO, r_min: float = R_MIN,
                    r_max: float = R_MAX, tol: float = SUM_TOL) -> list[float]:
    """Fail-closed check of a ratio vector; returns it as a list of floats."""
    if isinstance(ratios, (str, bytes, dict)) or not hasattr(ratios, "__iter__"):
        raise ValueError("layer ratios must be a sequence of numbers")
    values = list(ratios)
    if len(values) != int(num_layers):
        raise ValueError(f"expected {num_layers} layer ratios, got {len(values)}")
    if not all(_is_real(v) for v in values):
        raise ValueError("layer ratios must be finite real numbers")
    values = [float(v) for v in values]
    if not (_is_real(r_min) and _is_real(r_max) and 0 <= r_min <= r_max <= 1):
        raise ValueError("invalid clip range")
    if any(v < r_min - tol or v > r_max + tol for v in values):
        raise ValueError(f"layer ratio outside [{r_min}, {r_max}]")
    target = float(num_layers) * float(total_ratio)
    if abs(sum(values) - target) > tol:
        raise ValueError(f"layer ratios sum {sum(values)!r} != {target!r}")
    return values


def total_row_budget(n_content: int, num_layers: int = NUM_LAYERS,
                     total_ratio: float = TOTAL_RATIO) -> int:
    """``B(N) = L * ceil(total_ratio * N)``, the uniform store's row total."""
    if isinstance(n_content, bool) or not isinstance(n_content, Integral) \
            or n_content < 1:
        raise ValueError("n_content must be a positive integer")
    return int(num_layers) * math.ceil(float(total_ratio) * int(n_content))


def keep_bounds(n_content: int, r_min: float = R_MIN,
                r_max: float = R_MAX) -> tuple[int, int]:
    """Allowed ``[lo, hi]`` for ``k_l`` after integer rounding.

    Exact rational arithmetic on the decimal representation of the bounds,
    so ``ceil(0.6 * N)`` cannot drift by one through binary rounding.
    """
    from fractions import Fraction
    n = int(n_content)
    lo = math.floor(Fraction(repr(float(r_min))) * n)
    hi = min(n, math.ceil(Fraction(repr(float(r_max))) * n))
    return int(lo), int(hi)


def layer_keep_counts(ratios, n_content: int, num_layers: int = NUM_LAYERS,
                      total_ratio: float = TOTAL_RATIO, r_min: float = R_MIN,
                      r_max: float = R_MAX) -> list[int]:
    """Per-layer stored/visible content rows ``k_l`` for one image.

    Hamilton (largest-remainder) apportionment of ``B(N)`` with quotas
    ``r_l * N``.  Deterministic, exact-sum, capped at ``N`` per layer.
    """
    r = validate_ratios(ratios, num_layers, total_ratio, r_min, r_max)
    budget = total_row_budget(n_content, num_layers, total_ratio)
    n = int(n_content)
    quotas = [v * n for v in r]
    lo, hi = keep_bounds(n, r_min, r_max)
    counts = [min(hi, math.floor(q)) for q in quotas]
    remainders = [q - math.floor(q) for q in quotas]
    deficit = budget - sum(counts)
    if deficit < 0:
        raise ValueError("quota floors exceed the row budget")
    order = sorted(range(len(r)), key=lambda li: (-remainders[li], li))
    while deficit > 0:
        progressed = False
        for li in order:
            if deficit == 0:
                break
            # Extra rows never lift a layer past ceil(r_max * N) (or N).
            if counts[li] < hi:
                counts[li] += 1
                deficit -= 1
                progressed = True
        if not progressed:
            raise ValueError("row budget cannot be apportioned under the caps")
    assert sum(counts) == budget
    assert all(lo <= c <= hi for c in counts)
    return counts


def validate_keep_counts(counts, ratios, n_content: int,
                         num_layers: int = NUM_LAYERS,
                         total_ratio: float = TOTAL_RATIO,
                         r_min: float = R_MIN, r_max: float = R_MAX) -> None:
    """Recompute from ``ratios``/``N`` and require exact equality and bounds."""
    expected = layer_keep_counts(ratios, n_content, num_layers, total_ratio,
                                 r_min, r_max)
    if not isinstance(counts, (list, tuple)) or len(counts) != len(expected):
        raise ValueError("layer keep counts have the wrong length")
    if any(isinstance(c, bool) or not isinstance(c, Integral) for c in counts):
        raise ValueError("layer keep counts must be integers")
    if [int(c) for c in counts] != expected:
        raise ValueError("layer keep counts do not match the ratio vector")
    lo, hi = keep_bounds(n_content, r_min, r_max)
    if any(c < lo or c > hi for c in expected):
        raise ValueError(f"layer keep count outside [{lo}, {hi}]")
    if sum(expected) != total_row_budget(n_content, num_layers, total_ratio):
        raise ValueError("layer keep counts do not sum to the row budget")


def budget_sha256(ratios) -> str:
    """SHA256 of the canonical JSON encoding of the ratio vector."""
    payload = json.dumps([float(v) for v in ratios], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


# ------------------------------------------------------------ allocation
def clip_redistribute(raw, num_layers: int = NUM_LAYERS,
                      total_ratio: float = TOTAL_RATIO, r_min: float = R_MIN,
                      r_max: float = R_MAX, max_rounds: int = 1000) -> list[float]:
    """Scale ``raw >= 0`` to ``sum = L*total_ratio``, clip to ``[r_min, r_max]``
    and hand the residual proportionally to the still-unclipped layers.

    Repeats until no layer newly hits a bound.  Fails closed when every
    layer is clipped and the total cannot be met.
    """
    values = list(raw)
    if len(values) != int(num_layers) or not all(_is_real(v) and v >= 0 for v in values):
        raise ValueError("raw allocation must be finite, non-negative, one per layer")
    target = float(num_layers) * float(total_ratio)
    if not (r_min * num_layers <= target <= r_max * num_layers):
        raise ValueError("clip range cannot contain the total")
    if sum(values) <= 0:
        raise ValueError("raw allocation has no mass")
    L = len(values)
    fixed: list = [None] * L
    r = [0.0] * L
    for _ in range(max_rounds):
        free = [li for li in range(L) if fixed[li] is None]
        residual = target - sum(v for v in fixed if v is not None)
        if not free:
            if abs(residual) > SUM_TOL:
                raise ValueError("every layer clipped; total unreachable")
            break
        mass = sum(values[li] for li in free)
        for li in free:
            r[li] = (values[li] / mass * residual if mass > 0
                     else residual / len(free))
        for li in range(L):
            if fixed[li] is not None:
                r[li] = fixed[li]
        # Clip one bound per round so the redistribution can react: layers
        # above r_max first (their excess lifts the others), then layers
        # below r_min (their deficit is taken from the others).
        over = [li for li in free if r[li] > r_max + SUM_TOL]
        if over:
            for li in over:
                fixed[li] = r_max
            continue
        under = [li for li in free if r[li] < r_min - SUM_TOL]
        if under:
            for li in under:
                fixed[li] = r_min
            continue
        break
    else:
        raise ValueError("clip/redistribute did not converge")
    r = [min(r_max, max(r_min, v)) for v in r]
    drift = target - sum(r)
    free = [li for li in range(L) if fixed[li] is None]
    if abs(drift) > SUM_TOL:
        if not free:
            raise ValueError("every layer clipped; total unreachable")
        r[free[0]] += drift
    return validate_ratios(r, num_layers, total_ratio, r_min, r_max)


def allocate_power(sensitivity, gamma: float, **kw) -> list[float]:
    """A1: ``r_l`` proportional to ``max(s_l, 0) ** gamma`` then clipped."""
    if not (_is_real(gamma) and gamma > 0):
        raise ValueError("gamma must be a positive finite number")
    s = [max(0.0, float(v)) if _is_real(v) else None for v in sensitivity]
    if any(v is None for v in s):
        raise ValueError("sensitivity must be finite")
    if sum(s) <= 0:
        raise ValueError("sensitivity has no positive mass; A1 undefined")
    return clip_redistribute([v ** float(gamma) for v in s], **kw)


def allocate_water_filling(fractions, curves, step: float,
                           num_layers: int = NUM_LAYERS,
                           total_ratio: float = TOTAL_RATIO,
                           r_min: float = R_MIN, r_max: float = R_MAX) -> list[float]:
    """A2: greedy marginal-coverage water-filling on a shared fraction grid.

    ``curves[l][g]`` is the mean cumulative attention-mass coverage of the
    first ``fractions[g] * N`` stored rows in layer ``l`` (non-decreasing in
    ``g``, ``fractions[0] == 0``).  Every layer starts at ``r_min``; ``step``
    (a ratio, e.g. ``64 / median_N``) is handed to the layer with the largest
    marginal coverage gain until the total is reached; the last increment is
    truncated to land exactly on the total.
    """
    fr = [float(v) for v in fractions]
    if len(fr) < 2 or fr[0] != 0.0 or any(b <= a for a, b in zip(fr, fr[1:])):
        raise ValueError("fractions must start at 0 and strictly increase")
    cv = [[float(v) for v in row] for row in curves]
    if len(cv) != int(num_layers) or any(len(row) != len(fr) for row in cv):
        raise ValueError("curves must be (num_layers, len(fractions))")
    if not (_is_real(step) and step > 0):
        raise ValueError("step must be positive")

    def cov(li, f):
        row = cv[li]
        if f <= fr[0]:
            return row[0]
        if f >= fr[-1]:
            return row[-1]
        for g in range(1, len(fr)):
            if f <= fr[g]:
                a, b = fr[g - 1], fr[g]
                return row[g - 1] + (row[g] - row[g - 1]) * (f - a) / (b - a)
        return row[-1]

    target = float(num_layers) * float(total_ratio)
    r = [float(r_min)] * int(num_layers)
    remaining = target - sum(r)
    if remaining < -SUM_TOL:
        raise ValueError("r_min * L exceeds the total")
    while remaining > SUM_TOL:
        inc = min(float(step), remaining)
        best, gain = None, None
        for li in range(int(num_layers)):
            if r[li] + inc > r_max + SUM_TOL:
                continue
            g = cov(li, r[li] + inc) - cov(li, r[li])
            if best is None or g > gain + 1e-15:
                best, gain = li, g
        if best is None:
            raise ValueError("every layer at r_max before the total was reached")
        r[best] += inc
        remaining -= inc
    r = [min(r_max, max(r_min, v)) for v in r]
    return validate_ratios(r, num_layers, total_ratio, r_min, r_max)


def front_schedule(num_layers: int = NUM_LAYERS, r_first: float = 0.40,
                   r_last: float = 0.10, total_ratio: float = TOTAL_RATIO,
                   r_min: float = R_MIN, r_max: float = R_MAX,
                   reverse: bool = False) -> list[float]:
    """F-front: linear ``r_first -> r_last`` over layers, normalized to the
    total.  ``reverse=True`` gives F-back."""
    L = int(num_layers)
    raw = [r_first + (r_last - r_first) * li / (L - 1) for li in range(L)]
    if reverse:
        raw = raw[::-1]
    target = L * float(total_ratio)
    s = sum(raw)
    r = [v / s * target for v in raw]
    return validate_ratios(r, L, total_ratio, r_min, r_max)


# ------------------------------------------------------------- documents
def budget_document(alloc_id: str, ratios, *, model: str, source: dict,
                    num_layers: int = NUM_LAYERS, total_ratio: float = TOTAL_RATIO,
                    r_min: float = R_MIN, r_max: float = R_MAX) -> dict:
    r = validate_ratios(ratios, num_layers, total_ratio, r_min, r_max)
    return {
        "schema": "layer_budget_v1", "policy": POLICY, "alloc_id": str(alloc_id),
        "model": str(model), "num_layers": int(num_layers),
        "total_ratio": float(total_ratio), "r_min": float(r_min),
        "r_max": float(r_max), "layer_keep_ratios": r,
        "layer_budget_sha256": budget_sha256(r), "source": dict(source),
        "rounding": "hamilton_largest_remainder; floor then +1 by descending "
                    "fractional remainder (ties: lower layer first); cap N; "
                    "sum == L*ceil(total_ratio*N)",
    }


def load_budget_document(path) -> dict:
    from pathlib import Path
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "layer_budget_v1" or doc.get("policy") != POLICY:
        raise ValueError("not a layer_budget_v1 document")
    r = validate_ratios(doc["layer_keep_ratios"], doc["num_layers"],
                        doc["total_ratio"], doc["r_min"], doc["r_max"])
    if doc.get("layer_budget_sha256") != budget_sha256(r):
        raise ValueError("layer budget SHA256 mismatch")
    doc["layer_keep_ratios"] = r
    return doc

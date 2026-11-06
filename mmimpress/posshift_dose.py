"""PosShift dose-response: shift schedule, per-request checks and the pre-fixed reading.

Contract: docs/posshift_dose_contract.md.  Serving, stores and gates are the
unchanged recompute-KV25 ablation code (scripts/134, mmimpress/recompute_kv25);
this module only holds the pure helpers so they can be unit-tested.
"""
from __future__ import annotations
import torch

DELTAS = (0, 1, 4, 16, 64, 256, 1024)
COLLAPSE_EM = 0.05          # EM at or below this is "collapsed"
ABRUPT_MAX_DELTA = 4        # delta* <= 4 -> abrupt
GRADED_MIN_DELTA = 16       # delta* >= 16 -> graded


def shift_schedule(prefix_len, compact_start):
    """(delta, text_start, label) for the contract's deltas plus delta_max.

    ``prefix_len`` is v0+V (the original text start); ``compact_start`` is
    v0+k+n_sep.  delta_max is their difference.  Deltas that would not be
    smaller than delta_max are dropped (recorded by the caller).
    """
    dmax = int(prefix_len) - int(compact_start)
    assert dmax > 0, (prefix_len, compact_start)
    out = [(d, prefix_len - d, f"shift_{d}") for d in DELTAS if d < dmax]
    out.append((dmax, compact_start, "shift_max"))
    assert all(0 < s <= prefix_len for _, s, _ in out)
    return out, dmax


def bitwise_equal_logits(a, b):
    """True when two recorded per-step logit lists are equal step by step."""
    if a is None or b is None or len(a) != len(b):
        return False
    return all(torch.equal(x, y) for x, y in zip(a, b))


def tokens_equal(row, ref):
    return (row["first_token_id"] == ref["first_token_id"]
            and list(row["generated_token_ids"]) == list(ref["generated_token_ids"])
            and row["prediction"] == ref["prediction"])


def cap_reached(row, max_new_tokens=16, eos_id=2):
    ids = list(row["generated_token_ids"])
    return len(ids) >= max_new_tokens and eos_id not in ids


def reading(em_by_delta):
    """Pre-fixed reading from {delta: hit EM in [0,1]} (delta_max included)."""
    collapsed = sorted(d for d, em in em_by_delta.items() if em <= COLLAPSE_EM)
    if not collapsed:
        return {"delta_star": None, "reading": "not_reproduced",
                "text": "no delta (including delta_max) reaches EM <= 5%: the 061243Z PosShift result "
                        "is not reproduced; not interpreted before the cause is found"}
    d = collapsed[0]
    if d <= ABRUPT_MAX_DELTA:
        kind, text = "abrupt", "collapse within 1-4 positions: implementation to be checked first"
    elif d >= GRADED_MIN_DELTA:
        kind, text = "graded", "small shifts tolerated, collapse at a large shift: the 0% is read as a " \
                               "function of the size of the position mismatch"
    else:                      # unreachable with the contract's schedule; kept for completeness
        kind, text = "undefined", "delta* between 4 and 16 is not covered by the contract"
    return {"delta_star": int(d), "reading": kind, "text": text}


def monotone_non_increasing(points, tol=0.01):
    """Point estimates ordered by delta never rise by more than ``tol``."""
    return all(b <= a + tol for a, b in zip(points, points[1:]))

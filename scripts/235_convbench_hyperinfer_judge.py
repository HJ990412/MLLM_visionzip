#!/usr/bin/env python3
"""ConvBench HyperInfer run: the frozen judge applied to HyperInfer's answers (contract section 4).

Runs only for a model whose main-phase tokens are not all equal to the recorded IMPRESS tokens
(<res>/gates/<model>/identity.json, ``judge_needed``).  The judge is the one of the main experiment:
scripts/219 ``judge_model`` (which uses scripts/214 as it is) is called with this run's arm table, i.e. for the
single method HyperInfer, over all 577 conversations, with an EMPTY cache of its own (<res>/judge_cache);
no earlier judge cache is read.

  --res RES --model M    -> <res>/judge/<M>/{judgments/hyperinfer/<row>.json, summary.json, judge_raw.jsonl, DONE}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_hyperinfer_common as HC  # noqa: E402

C = HC.C


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True)
    p.add_argument("--model", choices=C.MODELS, required=True)
    args = p.parse_args()
    res = args.res.resolve()
    assert res.parent == ROOT / "results" and res.name.startswith(C.RUN_PREFIX)
    HC.verify_freeze(res)
    identity = C.read_json(res / "gates" / args.model / "identity.json")
    if identity is None:
        raise SystemExit("the comparison with the recorded IMPRESS tokens has not been made")
    if not identity["judge_needed"]:
        raise SystemExit("all tokens equal the recorded IMPRESS tokens: the recorded IMPRESS judgments are used; nothing to judge")
    J219 = HC.load_with_common("_convbench_hyperinfer_judge_base", "219_convbench_main_judge.py")
    assert C.MAIN_ROLES == [HC.HYPER]
    J219.judge_model(argparse.Namespace(res=res, model=args.model, cap=C.JUDGE_CAP, reuse_alignment_cache=False), J219.load_214())


if __name__ == "__main__":
    main()

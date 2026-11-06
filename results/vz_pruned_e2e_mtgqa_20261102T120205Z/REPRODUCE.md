# Reproduce / resume

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/vz_pruned_e2e_mtgqa_$(date -u +%Y%m%dT%H%M%SZ)
$PY scripts/147_eval_vz_pruned_e2e_mtgqa.py --run $RUN --freeze
$PY -m unittest tests.test_recompute_kv25 tests.test_rerope_kv25 tests.test_vz_pruned_e2e > $RUN/cpu_tests.log 2>&1
$PY scripts/147_eval_vz_pruned_e2e_mtgqa.py --run $RUN --protect before
$PY scripts/147_eval_vz_pruned_e2e_mtgqa.py --run $RUN --gpu correctness,smoke,integration,main   # add --resume after an interruption
$PY scripts/147_eval_vz_pruned_e2e_mtgqa.py --run $RUN --protect after
$PY scripts/149_report_vz_pruned_e2e_mtgqa.py --run $RUN --output results/$(basename $RUN)
```

This run: `/home/dblab/hj/mllm_v2/runs/vz_pruned_e2e_mtgqa_20261102T120205Z`. Frozen sources and their SHA256 are in `gpu_freeze.json`; the contract copy is `CONTRACT.md`;
the pre-freeze development check (not a result) is in `dev/`. Stores of smoke/integration/main images were deleted
after audit and commit (sealed hash lists in each image's `cleanup_receipt.json`); the five G4 images keep their
stores under `scratch/correctness/`. Reproducing an image's payload requires re-running its source dialogue's T1.
The report script is excluded from the freeze; 148 (audit) is frozen and re-executed at report time.

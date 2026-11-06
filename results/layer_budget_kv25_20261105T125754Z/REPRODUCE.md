# Reproduce

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=/home/dblab/hj/mllm_v2/runs/layer_budget_kv25_20261105T125754Z
# Phase 0/1: data split, N distribution, storage plan (see run/phase0), contract freeze (config.json, phase1_freeze.json)
$PY scripts/152_profile_layer_sensitivity.py --run $RUN --max-images 10 --probe-report $RUN/phase1/timing_probe.json
$PY scripts/152_profile_layer_sensitivity.py --run $RUN
$PY scripts/153_allocate_layer_budget.py --run $RUN
$PY scripts/154_validate_layer_budget_kv25.py --run $RUN --stage cpu
$PY scripts/154_validate_layer_budget_kv25.py --run $RUN --stage g3
$PY scripts/154_validate_layer_budget_kv25.py --run $RUN --stage gpu
$PY scripts/155_eval_layer_budget_kv25.py --run $RUN --freeze
$PY scripts/155_eval_layer_budget_kv25.py --run $RUN --gpu smoke
$PY scripts/155_eval_layer_budget_kv25.py --run $RUN --gpu gqa
$PY scripts/155_eval_layer_budget_kv25.py --run $RUN --analyze
$PY scripts/156_eval_layer_budget_hold.py --run $RUN --stage freeze
$PY scripts/156_eval_layer_budget_hold.py --run $RUN --stage smoke
$PY scripts/156_eval_layer_budget_hold.py --run $RUN --stage main        # --resume after an interruption
$PY scripts/157_report_layer_budget_kv25.py --run $RUN
```
A new reproduction must use a new run root; existing runs/results/stores are never modified. HOLD payloads are deleted after each image's audit and commit and must be regenerated to re-serve.

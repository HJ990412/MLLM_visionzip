# Reproduce (v2)

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=/home/dblab/hj/mllm_v2/runs/layer_budget_kv25_v2_20261106T073502Z
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage freeze     # Phase 0 (contract SHA, v1 reuse, storage plan, protection before)
$PY scripts/158_allocate_prior_work_vectors.py --run $RUN                 # Phase 1 vectors + G8
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage g9          # dd microbench
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage cpu
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage g3
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage gpu         # G1/G2/G4 + G6/G7
$PY scripts/159_eval_layer_budget_v2_dev.py --run $RUN --freeze
$PY scripts/159_eval_layer_budget_v2_dev.py --run $RUN --gpu smoke
$PY scripts/159_eval_layer_budget_v2_dev.py --run $RUN --gpu gqa
$PY scripts/159_eval_layer_budget_v2_dev.py --run $RUN --analyze
$PY scripts/162_eval_layer_budget_v2_hold.py --run $RUN --stage freeze
$PY scripts/162_eval_layer_budget_v2_hold.py --run $RUN --stage smoke
$PY scripts/162_eval_layer_budget_v2_hold.py --run $RUN --stage main       # --resume after an interruption
$PY scripts/160_report_layer_budget_v2.py --run $RUN
```
A new reproduction must use a new run root; existing runs/results/stores are never modified. HOLD payloads are deleted after each image's audit and commit (run-local scratch only) and must be regenerated to re-serve.

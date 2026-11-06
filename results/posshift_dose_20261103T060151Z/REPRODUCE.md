# Reproduce

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/posshift_dose_$(date -u +%Y%m%dT%H%M%SZ)
$PY scripts/150_eval_posshift_dose.py --run $RUN --freeze
$PY -m unittest tests.test_recompute_kv25 tests.test_posshift_dose > $RUN/cpu_tests.log 2>&1
$PY scripts/150_eval_posshift_dose.py --run $RUN --protect before
$PY scripts/150_eval_posshift_dose.py --run $RUN --gpu smoke,gqa
$PY scripts/150_eval_posshift_dose.py --run $RUN --protect after
$PY scripts/151_report_posshift_dose.py --run $RUN
```

This run: `/home/dblab/hj/mllm_v2/runs/posshift_dose_20261103T060151Z`. Raw rows `llava/gqa/raw_requests.jsonl` (one row per hit, with `delta`, `text_position_start`, `position_check`, `delta0_bitwise_equal_base`, `reproduces_prior`); per-image schedule and gate receipts `llava/gqa/gates.jsonl`; persistence receipts `llava/gqa/persistence.jsonl`. Stores are deleted after each image (`llava/cleanup.jsonl`). Prior rows come from `runs/recompute_kv25_ablation_20261102T061243Z/llava/gqa/raw_requests.jsonl` (sha256 `0f6c839a8062c5a8d03ec48417263af626fe82481d5ca40b279deaf4506c34e9`, read only). The report script (151) is outside the freeze.

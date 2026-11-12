# Reproduce (PyramidKV beta=20 HOLD add-on)

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/layer_budget_kv25_pyrbeta_<new timestamp>
$PY scripts/163_make_pyrbeta_raw_vector.py --run $RUN --stage vectors      # copy U / PYR-beta (SHA check), build PYR-beta-raw, k_l tables
# write / check docs/layer_budget_kv25_pyrbeta_contract.md, then:
$PY scripts/163_make_pyrbeta_raw_vector.py --run $RUN --stage freeze       # contract SHA, storage plan, protection before, hold_arms.json
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage g3
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage grmin
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage cpu
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage gpu     # G6/G7, five fixed cases, both vectors
$PY scripts/164_eval_layer_budget_pyrbeta_hold.py --run $RUN --stage freeze
$PY scripts/164_eval_layer_budget_pyrbeta_hold.py --run $RUN --stage smoke
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage gdet --phase smoke
$PY scripts/164_eval_layer_budget_pyrbeta_hold.py --run $RUN --stage main   # --resume after an interruption
$PY scripts/165_report_layer_budget_pyrbeta.py --run $RUN                   # audit, G-det main, decisions, cross-run, protection after
```
A new reproduction must use a new run root; existing runs/results/stores are never modified. HOLD payloads are deleted after each
image's audit and commit (run-local scratch only) and must be regenerated to re-serve. The optional recompute arm is added with
`--with-recompute` at the 163 freeze stage (only when allowed before execution).

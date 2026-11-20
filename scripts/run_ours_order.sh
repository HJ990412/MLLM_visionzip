#!/bin/bash
# LLaVA-NeXT x MT-GQA, Ours(재배열) vs Ours(원래 순서): the whole chain (contract docs/mt_gqa_ours_order_contract.md).
#   scripts/run_ours_order.sh runs/mt_gqa_ours_order_<UTC> [--resume-main]
# freeze -> gate 6 (cpu) -> gate phase (4 images) -> gates 1/3/4/5 -> gate 2 (4 images)
#        -> devgate phase (DEV 40) -> gates 3/5 -> gate 2 (DEV 40) -> dev pilot (DEV 40) -> gates 1/4/7
#        -> admit -> main -> posthoc -> report.
# A failed gate skips the remaining GPU phases; admission and the report are still written.
cd /home/dblab/hj/mllm_v2 || exit 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
CROSS=/home/dblab/hj/mllm_v2/.venv-glm46v/bin/python
RUN=$1; MODE=${2:-fresh}
EVAL=scripts/202_eval_mt_gqa_ours_order.py; VAL=scripts/203_validate_mt_gqa_ours_order.py; REP=scripts/204_report_mt_gqa_ours_order.py
RES=results/$(basename "$RUN")
ssd() { echo "$(date -u +%FT%TZ) $(cat /sys/block/nvme0n1/stat)"; }
step() {   # step <name> <command...>: appends to the chain log, returns the command's status
  local name=$1; shift
  echo "=== $(date -u +%FT%TZ) START $name" >> "$LOG"
  "$@" >> "$LOG" 2>&1; local rc=$?
  echo "=== $(date -u +%FT%TZ) END $name rc=$rc" >> "$LOG"
  return $rc
}

if [ "$MODE" = "--resume-main" ]; then
  LOG=$RUN/validation_chain.log
  step main_resume $PY $EVAL --run "$RUN" --stage main --resume || { echo "after_main_failed $(ssd)" >> "$RUN/ssd_stat.txt"; exit 1; }
else
  [ -e "$RUN" ] && { echo "run directory exists: $RUN"; exit 2; }
  BEFORE=$(ssd)
  TMP=$(mktemp)
  $PY $EVAL --run "$RUN" --stage freeze > "$TMP" 2>&1 || { cat "$TMP"; rm -f "$TMP"; exit 1; }
  LOG=$RUN/validation_chain.log
  { echo "=== freeze"; cat "$TMP"; } >> "$LOG"; rm -f "$TMP"
  echo "before_freeze $BEFORE" > "$RUN/ssd_stat.txt"
  mkdir -p "$RES"
  $PY - "$RUN" "$RES" "$BEFORE" <<'EOF'
import json, sys, time, pathlib
run, res, before = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
plan = json.loads((run / "storage_plan.json").read_text())
doc = {"written_before_any_gpu_measurement_unix": time.time(), "free_bytes_at_freeze": plan["free_bytes_at_freeze"],
       "free_gib_at_freeze": plan["free_bytes_at_freeze"] / 2**30, "required_free_bytes": plan["required_free_bytes"],
       "expected_ssd_write_bytes": plan["expected_ssd_write_bytes"], "expected_ssd_write_note": plan["expected_ssd_write_note"],
       "nvme0n1_stat_before": before, "nvme0n1_sectors_written_before": int(before.split()[7]), "sector_bytes": 512}
(res / "prerun.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
EOF
  OK=1
  step cpu $PY $VAL --run "$RUN" --stage cpu --extra-python $CROSS || OK=0
  if step gate_phase $PY $EVAL --run "$RUN" --stage gate; then
    step gate_checks $PY $VAL --run "$RUN" --stage gate || OK=0
    step gate2_gate $PY $VAL --run "$RUN" --stage qindep --phase gate || OK=0
  else OK=0; fi
  echo "after_gate $(ssd)" >> "$RUN/ssd_stat.txt"
  if [ $OK = 1 ]; then
    if step devgate_phase $PY $EVAL --run "$RUN" --stage devgate; then
      step devgate_checks $PY $VAL --run "$RUN" --stage devgate || OK=0
      step gate2_dev $PY $VAL --run "$RUN" --stage qindep --phase devgate || OK=0
    else OK=0; fi
    echo "after_devgate $(ssd)" >> "$RUN/ssd_stat.txt"
  fi
  if [ $OK = 1 ]; then
    if step dev_phase $PY $EVAL --run "$RUN" --stage dev; then
      step dev_checks $PY $VAL --run "$RUN" --stage dev || OK=0
    else OK=0; fi
    echo "after_dev $(ssd)" >> "$RUN/ssd_stat.txt"
  fi
  step admit $PY $VAL --run "$RUN" --stage admit || OK=0
  if [ $OK = 1 ]; then
    echo $$ > "$RUN/main.pid"
    step main_phase $PY $EVAL --run "$RUN" --stage main || OK=0
  fi
fi
echo "after_main $(ssd)" >> "$RUN/ssd_stat.txt"
if [ -e "$RUN/main_execution_complete.json" ]; then step posthoc $PY $VAL --run "$RUN" --stage posthoc; fi
step report $PY $REP --run "$RUN"
echo "=== $(date -u +%FT%TZ) CHAIN END" >> "$LOG"
[ -e "$RUN/main_execution_complete.json" ]

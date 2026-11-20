#!/bin/bash
# Experiment A (LLaVA-NeXT x MT-GQA, IMPRESS = public-implementation path): the whole chain.
#   scripts/run_impress_code_A.sh runs/mt_gqa_impress_code_<UTC> [--resume-main]
# freeze -> step 1/6 (cpu) -> gate -> steps 3/4/5 -> step 2 (prefix) -> dev -> step 7 -> admit -> main -> posthoc -> report.
# A failed validation step skips the remaining GPU phases; admission and the report are still written.
cd /home/dblab/hj/mllm_v2 || exit 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export IMPRESS_OFFICIAL_REPO=/home/dblab/hj/_external_ref/HyperInfer
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
CROSS=/home/dblab/hj/mllm_v2/.venv-glm46v/bin/python
RUN=$1; MODE=${2:-fresh}
EVAL=scripts/196_eval_mt_gqa_impress_code.py; VAL=scripts/197_validate_mt_gqa_impress_code.py; REP=scripts/198_report_mt_gqa_impress_code.py
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
  step main_resume $PY $EVAL --run "$RUN" --stage main --resume || { echo "$(ssd)" >> "$RUN/ssd_stat.txt"; exit 1; }
else
  [ -e "$RUN" ] && { echo "run directory exists: $RUN"; exit 2; }
  BEFORE=$(ssd)
  $PY $EVAL --run "$RUN" --stage freeze > /tmp/freeze_A_$$.log 2>&1 || { cat /tmp/freeze_A_$$.log; exit 1; }
  LOG=$RUN/validation_chain.log
  { echo "=== freeze"; cat /tmp/freeze_A_$$.log; } >> "$LOG"; rm -f /tmp/freeze_A_$$.log
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
    step prefix $PY $VAL --run "$RUN" --stage prefix || OK=0
  else OK=0; fi
  echo "after_gate $(ssd)" >> "$RUN/ssd_stat.txt"
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

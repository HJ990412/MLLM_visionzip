#!/bin/bash
# Experiment B (LLaVA-1.5 x MT-VQA 25k, IMPRESS = public-implementation path): the whole chain.
#   scripts/run_impress_code_B.sh runs/llava15_mtvqa_impress_code_<UTC> [--resume-main]
# freeze -> step 1/6 (cpu) -> gate -> steps 3/4/5 -> step 2 (prefix) -> pilot -> step 7 -> admit -> main -> posthoc -> report.
# A failed validation step skips the remaining GPU phases; admission and the report are still written.
cd /home/dblab/hj/mllm_v2 || exit 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export IMPRESS_OFFICIAL_REPO=/home/dblab/hj/_external_ref/HyperInfer
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=$1; MODE=${2:-fresh}
EVAL=scripts/199_eval_llava15_mtvqa_impress_code.py; VAL=scripts/200_validate_llava15_mtvqa_impress_code.py; REP=scripts/201_report_llava15_mtvqa_impress_code.py
RES=results/$(basename "$RUN")
ssd() { echo "$(date -u +%FT%TZ) $(cat /sys/block/nvme0n1/stat)"; }
step() {
  local name=$1; shift
  echo "=== $(date -u +%FT%TZ) START $name" >> "$LOG"
  "$@" >> "$LOG" 2>&1; local rc=$?
  echo "=== $(date -u +%FT%TZ) END $name rc=$rc" >> "$LOG"
  return $rc
}

if [ "$MODE" = "--resume-main" ]; then
  LOG=$RUN/validation_chain.log
  echo $$ > "$RUN/main.pid"
  $PY $EVAL --run "$RUN" --stage main --resume >> "$RUN/main.log" 2>&1 || { echo "after_main_interrupted $(ssd)" >> "$RUN/ssd_stat.txt"; exit 1; }
else
  [ -e "$RUN" ] && { echo "run directory exists: $RUN"; exit 2; }
  BEFORE=$(ssd)
  $PY $EVAL --run "$RUN" --stage freeze > /tmp/freeze_B_$$.log 2>&1 || { cat /tmp/freeze_B_$$.log; exit 1; }
  LOG=$RUN/validation_chain.log
  { echo "=== freeze"; cat /tmp/freeze_B_$$.log; } >> "$LOG"; rm -f /tmp/freeze_B_$$.log
  echo "before_freeze $BEFORE" > "$RUN/ssd_stat.txt"
  mkdir -p "$RES"
  $PY - "$RUN" "$RES" "$BEFORE" <<'EOF'
import json, sys, time, pathlib
run, res, before = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
plan = json.loads((run / "storage_plan.json").read_text())
doc = {"written_before_any_gpu_measurement_unix": time.time(), "free_bytes_at_freeze": plan["free_bytes_at_freeze"],
       "free_gib_at_freeze": plan["free_bytes_at_freeze"] / 2**30, "required_free_bytes": plan["required_free_bytes"],
       "expected_total_ssd_write_bytes": plan["expected_total_ssd_write_bytes"], "expected_total_ssd_write_note": plan["expected_total_ssd_write_note"],
       "per_image_bundle_bytes": plan["per_image_bundle_bytes"],
       "nvme0n1_stat_before": before, "nvme0n1_sectors_written_before": int(before.split()[7]), "sector_bytes": 512}
(res / "prerun.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
EOF
  OK=1
  step cpu $PY $VAL --run "$RUN" --stage cpu || OK=0
  if step gate_phase $PY $EVAL --run "$RUN" --stage gate; then
    step gate_checks $PY $VAL --run "$RUN" --stage gate || OK=0
    step prefix $PY $VAL --run "$RUN" --stage prefix || OK=0
  else OK=0; fi
  echo "after_gate $(ssd)" >> "$RUN/ssd_stat.txt"
  if [ $OK = 1 ]; then
    if step pilot_phase $PY $EVAL --run "$RUN" --stage pilot; then
      step pilot_checks $PY $VAL --run "$RUN" --stage pilot || OK=0
      step pilot_report $PY $REP --run "$RUN" --phase pilot
    else OK=0; fi
    echo "after_pilot $(ssd)" >> "$RUN/ssd_stat.txt"
  fi
  step admit $PY $VAL --run "$RUN" --stage admit || OK=0
  if [ $OK = 1 ]; then
    echo $$ > "$RUN/main.pid"
    echo "=== $(date -u +%FT%TZ) START main_phase (output in main.log)" >> "$LOG"
    $PY $EVAL --run "$RUN" --stage main >> "$RUN/main.log" 2>&1; rc=$?
    echo "=== $(date -u +%FT%TZ) END main_phase rc=$rc" >> "$LOG"
  fi
fi
echo "after_main $(ssd)" >> "$RUN/ssd_stat.txt"
NOTES=""; [ -e "$RES/notes.md" ] && NOTES="--notes $RES/notes.md"
if [ -e "$RUN/main_execution_complete.json" ]; then
  step posthoc $PY $VAL --run "$RUN" --stage posthoc
  step report $PY $REP --run "$RUN" --phase main $NOTES
else
  step report $PY $REP --run "$RUN" --phase validation $NOTES
fi
echo "=== $(date -u +%FT%TZ) CHAIN END" >> "$LOG"
[ -e "$RUN/main_execution_complete.json" ]

#!/bin/bash
# Experiment A, then experiment B (B runs even if A fails).  Detached use:
#   setsid nohup scripts/run_impress_code_both.sh <master log> > /dev/null 2>&1 &
# Between the two, waits (at most 45 minutes) for results/<A>/NOTES_DONE so that A's
# hand-written notes and final report are in place before B freezes its protection inventory.
cd /home/dblab/hj/mllm_v2 || exit 2
MASTER=${1:-/tmp/impress_code_both.log}
A=runs/mt_gqa_impress_code_$(date -u +%Y%m%dT%H%M%SZ)
echo "$(date -u +%FT%TZ) A start $A pid $$" >> "$MASTER"
scripts/run_impress_code_A.sh "$A" >> "$MASTER" 2>&1
echo "$(date -u +%FT%TZ) A end rc=$? $A" >> "$MASTER"
for i in $(seq 1 270); do
  [ -e "results/$(basename "$A")/NOTES_DONE" ] && break
  sleep 10
done
echo "$(date -u +%FT%TZ) notes wait over (NOTES_DONE: $([ -e "results/$(basename "$A")/NOTES_DONE" ] && echo yes || echo no))" >> "$MASTER"
B=runs/llava15_mtvqa_impress_code_$(date -u +%Y%m%dT%H%M%SZ)
echo "$(date -u +%FT%TZ) B start $B" >> "$MASTER"
scripts/run_impress_code_B.sh "$B" >> "$MASTER" 2>&1
echo "$(date -u +%FT%TZ) B end rc=$? $B" >> "$MASTER"

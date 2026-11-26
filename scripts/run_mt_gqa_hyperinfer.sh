#!/bin/bash
# MT-GQA x {LLaVA-NeXT, LLaVA-1.5}: HyperInfer -- the whole chain, detached.
# Contract: docs/mt_gqa_hyperinfer_contract.md.
#
#   scripts/run_mt_gqa_hyperinfer.sh runs/mt_gqa_hyperinfer_<UTC>
#
# The same command starts the run and, after an interruption or a reboot, continues it.
# It re-executes itself in a new session (setsid + nohup), so it does not depend on the terminal,
# the SSH connection or the session that typed it; it prints where the PID file is and returns at once.
#
#   chunk read time calibration (both models; skipped when recorded) -> constants into the contract
#   -> freeze -> gate 7 (tests) -> gate 1 (logic against the public code)
#   -> LLaVA-NeXT: gate phase (4 images, 7 arms) -> gates 2, 3, 5, 6
#                  -> dev phase (DEV 40, 7 arms) -> gates 2, 3, 4, 5, 6 (+ gate 8, reported)
#                  -> admission -> main (398 images, HyperInfer)
#   -> LLaVA-1.5 : the same
#   -> posthoc -> report
#
# Every stage decides by itself; nothing waits for a person.  A stage that already has its result is
# skipped; a request phase that was interrupted continues (--resume: complete dialogues are kept, the
# interrupted dialogue restarts at turn 1).  If a gate of a model fails, that model's main phase is not
# started and the failure is written to status.json and report.md; the other model is still run.
#
# Files (results/<run name>/): status.json, driver.log, driver.pid, driver.lock, prerun.json, report.md
#   progress : cat results/<run name>/status.json
#   stop     : kill -TERM -- -$(cat results/<run name>/driver.pid)
#   continue : this same command

SELF=$(readlink -f "$0")
cd /home/dblab/hj/mllm_v2 || exit 2
RUN=${1:?usage: scripts/run_mt_gqa_hyperinfer.sh runs/mt_gqa_hyperinfer_<UTC>}
RUN=${RUN%/}
NAME=$(basename "$RUN")
case "$RUN" in runs/mt_gqa_hyperinfer_*) ;; *) echo "run must be runs/mt_gqa_hyperinfer_<UTC> (relative to the repository)"; exit 2;; esac
RES=results/$NAME
mkdir -p "$RES"

# ---------------------------------------------------------------- detach
if [ "${HYPERINFER_DETACHED:-0}" != "1" ]; then
  HYPERINFER_DETACHED=1 setsid nohup "$SELF" "$RUN" >> "$RES/driver.log" 2>&1 < /dev/null &
  echo "driver started in a new session (pid file $RES/driver.pid, log $RES/driver.log, status $RES/status.json)"
  exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export IMPRESS_OFFICIAL_REPO=/home/dblab/hj/_external_ref/HyperInfer
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
EVAL=scripts/225_eval_mt_gqa_hyperinfer.py; VAL=scripts/226_validate_mt_gqa_hyperinfer.py; REP=scripts/227_report_mt_gqa_hyperinfer.py
ST=scripts/llava_next_main_status.py
MODELS="next l15"
declare -A DIR=([next]=llava_next [l15]=llava15)
declare -A LABEL=([next]=LLaVA-NeXT-7B [l15]=LLaVA-1.5-7B)
MAX_PHASE_STARTS=${HYPERINFER_MAX_PHASE_STARTS:-3}      # automatic restarts of an interrupted request phase inside one driver run
GPU_WAIT_POLLS=${HYPERINFER_GPU_WAIT_POLLS:-480}        # x 30 s: how long a GPU stage waits for another process to leave the GPU

# ------------------------------------------------------------------ lock
exec 9> "$RES/driver.lock"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) another driver holds $RES/driver.lock (pid $(cat "$RES/driver.pid" 2>/dev/null)); this one exits"
  exit 3
fi
echo $$ > "$RES/driver.pid"

log() { echo "=== $(date -u +%FT%TZ) $*"; }
status() { $PY $ST "$NAME" "$@"; }
ssd() { [ -e "$RUN/config.json" ] && $PY $ST "$RUN" --ssd-sample "$1"; }
report() {
  if [ -e "$RUN/config.json" ]; then
    log "START report"; $PY $REP --run "$RUN"; local rc=$?; log "END report rc=$rc"
    [ $rc = 0 ] && status report="$RES/report.md" || status "last_error={\"stage\": \"report\", \"error\": \"report script rc=$rc (see driver.log)\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"
  fi
}
fail() {   # fail <stage> <message>: record and stop; the report is still written
  log "STOP $1: $2"
  status stage="stopped" stopped_at_stage="$1" "last_error={\"stage\": \"$1\", \"error\": \"$2\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"
  ssd "driver_stop"; report
  log "DRIVER END rc=1"
  exit 1
}
on_signal() {
  log "signal received: stopping"
  status stage="stopped_by_signal" "last_error={\"stage\": \"signal\", \"error\": \"driver received TERM/INT; continue with the same command\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"
  exit 130
}
trap on_signal TERM INT

log "DRIVER START pid $$ ppid $PPID sid $(ps -o sid= -p $$ | tr -d ' ') pgid $(ps -o pgid= -p $$ | tr -d ' ') tty $(ps -o tty= -p $$ | tr -d ' ') boot $(cat /proc/sys/kernel/random/boot_id)"
status driver_pid=$$ driver_started_utc="$(date -u +%FT%TZ)" last_error=null \
       "commands={\"progress\": \"cat $RES/status.json\", \"stop\": \"kill -TERM -- -\$(cat $RES/driver.pid)\", \"continue\": \"scripts/run_mt_gqa_hyperinfer.sh $RUN\"}"

wait_gpu_free() {   # a GPU stage starts only when no other process computes on the GPU (the harness aborts an image otherwise)
  local i=0 n
  while [ $i -lt "$GPU_WAIT_POLLS" ]; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "$n" = 0 ] && return 0
    [ $i = 0 ] && { log "GPU is used by another process; waiting"; status stage="waiting_for_gpu"; }
    sleep 30; i=$((i + 1))
  done
  return 1
}

# ------------------------------------------- calibration, constants, freeze
if [ ! -e "$RUN/config.json" ]; then
  for m in $MODELS; do
    if [ ! -e "$RES/calibration_$m.json" ]; then
      wait_gpu_free || fail "calibrate_$m" "the GPU stayed in use by another process"
      status stage="calibrate:$m" model="${LABEL[$m]}"
      log "START calibrate $m"; $PY $EVAL --run "$RUN" --stage calibrate --model "$m"; RC=$?; log "END calibrate $m rc=$RC"
      [ -e "$RES/calibration_$m.json" ] || fail "calibrate_$m" "the chunk read time calibration did not produce a result (rc=$RC; see driver.log)"
    else
      log "SKIP calibrate $m (recorded)"
    fi
  done
  status stage="constants"
  log "START constants"; $PY $EVAL --run "$RUN" --stage constants; RC=$?; log "END constants rc=$RC"
  [ $RC = 0 ] || fail constants "writing the measured constants into the contract failed (rc=$RC)"
  status stage="freeze"
  log "START freeze"; $PY $EVAL --run "$RUN" --stage freeze; RC=$?; log "END freeze rc=$RC"
  [ $RC = 0 ] && [ -e "$RUN/protection_before.json" ] || {
    [ -d "$RUN" ] && mv "$RUN" "${RUN}.incomplete_freeze_$(date -u +%Y%m%dT%H%M%SZ)"
    fail freeze "freeze rc=$RC (see driver.log)"; }
  $PY $ST "$RUN" --smart before
  KIND=fresh
else
  KIND=resume
fi
$PY $ST "$RUN" --invocation $$ $KIND
for m in $MODELS; do rm -f "$RUN/${DIR[$m]}/STOP_REQUESTED" "$RUN/${DIR[$m]}"/NO_RETRY_*.json; done
ssd "driver_start_$KIND"
status stage="starting" driver_kind="$KIND" driver_invocations="$(wc -l < "$RUN/driver_invocations.jsonl")"

gate_status() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$RUN/gates/$1.json" 2>/dev/null || echo MISSING; }

check() {   # check <gate file name> <stage args...>: run a gate stage once; a recorded result is never re-judged
  local name=$1; shift
  if [ "$(gate_status "$name")" = MISSING ]; then
    wait_gpu_free || fail "$name" "the GPU stayed in use by another process"
    status stage="$name"
    log "START $name"; $PY $VAL --run "$RUN" "$@"; log "END $name rc=$? status=$(gate_status "$name")"
    ssd "after_$name"
  else
    log "SKIP $name (recorded: $(gate_status "$name"))"
  fi
  # no result at all = the check itself broke (not a judged failure): stop, so that the same command retries it
  [ "$(gate_status "$name")" = MISSING ] && fail "$name" "the check did not produce a result (see driver.log)"
  [ "$(gate_status "$name")" = PASS ] || { status "gate_${name}=FAIL"; return 1; }
  status "gate_${name}=PASS"
}

phase() {   # phase <model> <name>: run or continue a request phase until it is complete
  local m=$1 ph=$2 n=0 rc=0 sub="$RUN/${DIR[$1]}"
  if [ -e "$sub/${ph}_execution_complete.json" ]; then log "SKIP $m ${ph} phase (complete)"; return 0; fi
  while [ $n -lt "$MAX_PHASE_STARTS" ]; do
    n=$((n + 1))
    local resume=""; [ -d "$sub/$ph" ] && resume="--resume"
    wait_gpu_free || return 3
    status stage="$m:${ph}_phase" model="${LABEL[$m]}"
    log "START $m ${ph} phase $resume (start $n of at most $MAX_PHASE_STARTS in this driver run)"
    $PY $EVAL --run "$RUN" --model "$m" --stage "$ph" $resume; rc=$?
    log "END $m ${ph} phase rc=$rc"
    ssd "after_${m}_${ph}_phase"
    [ -e "$sub/${ph}_execution_complete.json" ] && return 0
    [ -e "$sub/STOP_REQUESTED" ] && { log "STOP_REQUESTED found"; return 2; }
    # an error that would repeat (token ids, image audit, storage, frozen files): no automatic restart
    [ -e "$sub/NO_RETRY_${ph}.json" ] && { log "the $m ${ph} phase stopped on an error that a restart would repeat ($sub/NO_RETRY_${ph}.json)"; return 4; }
    sleep 60
  done
  return 1
}

# ------------------------------------------------- gates common to both models
COMMON=1
check tests --stage tests || COMMON=0
check logic --stage logic || COMMON=0
if [ $COMMON = 0 ]; then
  fail common_gates "gate 7 (existing tests) or gate 1 (logic against the public code) failed; no model is measured (details: runs/$NAME/gates/tests.json, logic.json)"
fi

# ------------------------------------------------------------ per model
ALL=0
for m in $MODELS; do
  if [ "$(gate_status "${m}_admission")" = MISSING ]; then
    OK=1
    if phase "$m" gate; then
      check "${m}_gate" --stage gate --model "$m" || OK=0
    else fail "${m}_gate_phase" "the $m gate phase did not complete (see driver.log and runs/$NAME/${DIR[$m]}/interruption_gate_*.json); continue with: scripts/run_mt_gqa_hyperinfer.sh $RUN"; fi
    if [ $OK = 1 ]; then
      if phase "$m" dev; then
        check "${m}_dev" --stage dev --model "$m" || OK=0
      else fail "${m}_dev_phase" "the $m dev phase did not complete (see driver.log and runs/$NAME/${DIR[$m]}/interruption_dev_*.json); continue with: scripts/run_mt_gqa_hyperinfer.sh $RUN"; fi
    fi
    status stage="$m:admission"
    log "START $m admission"; $PY $VAL --run "$RUN" --stage admit --model "$m"; log "END $m admission rc=$? status=$(gate_status "${m}_admission")"
  fi
  if [ "$(gate_status "${m}_admission")" != PASS ]; then
    log "$m: gates not passed; its main phase is not started (runs/$NAME/gates/${m}_admission.json)"
    status "admission_$m=FAIL" "main_$m=not started" "last_error={\"stage\": \"${m}_admission\", \"error\": \"gates of $m not passed; its main phase is not started (details: runs/$NAME/gates/${m}_admission.json and report.md)\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"
    ALL=1
    continue
  fi
  status "admission_$m=PASS" "main_$m=admitted"
  phase "$m" main; RC=$?
  if [ $RC != 0 ]; then
    fail "${m}_main_phase" "the $m main phase stopped before completion (rc=$RC; see runs/$NAME/${DIR[$m]}/interruption_main_*.json); continue with: scripts/run_mt_gqa_hyperinfer.sh $RUN"
  fi
  status "main_$m=complete"
done

# ------------------------------------------------------- posthoc and report
ssd "driver_finish"
log "START posthoc"; $PY $VAL --run "$RUN" --stage posthoc; log "END posthoc rc=$?"
$PY $ST "$RUN" --smart after
report
if [ $ALL = 0 ]; then
  status stage="done" finished_utc="$(date -u +%FT%TZ)" eta_utc="$(date -u +%FT%TZ)" eta_note="finished" driver_pid=null
else
  status stage="done_with_gate_failure" finished_utc="$(date -u +%FT%TZ)" eta_utc="$(date -u +%FT%TZ)" eta_note="finished; at least one model was not admitted to its main phase" driver_pid=null
fi
log "DRIVER END rc=$ALL"
exit $ALL

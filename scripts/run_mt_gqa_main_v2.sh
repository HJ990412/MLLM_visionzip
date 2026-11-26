#!/bin/bash
# LLaVA-NeXT x MT-GQA main experiment v2 (original-LLaVA token sequence; ReComp / FullLoad / IMPRESS / Ours):
# the whole chain, detached.  Contract: docs/mt_gqa_main_v2_contract.md.
#
#   scripts/run_mt_gqa_main_v2.sh runs/mt_gqa_main_v2_<UTC>
#
# The same command starts the run and, after an interruption or a reboot, continues it.
# It re-executes itself in a new session (setsid + nohup), so it does not depend on the terminal,
# the SSH connection or the session that typed it; it prints where the PID file is and returns at once.
#
#   freeze -> gate 7 (tests) -> gates 1, 2 (token ids, CPU)
#          -> legacy phase (4 images, earlier tokenization) -> gate 3
#          -> gate phase (4 images, 9 arms) -> gates 4, 5, 6
#          -> devgate phase (DEV 40, 7 arms) -> gates 4, 5
#          -> dev pilot (DEV 40, 4 arms) -> gates 6, 8 + pilot table
#          -> admission -> main (398 images, 4 arms) -> posthoc -> report
#
# Every stage decides by itself; nothing waits for a person.  A stage that already has its result is
# skipped; a request phase that was interrupted continues (--resume: complete dialogues are kept, the
# interrupted dialogue restarts at turn 1).  If a gate fails, the remaining GPU stages and the main phase
# are not started, the failure is written to status.json and report.md, and the driver exits.
#
# Files (results/<run name>/): status.json, driver.log, driver.pid, driver.lock, prerun.json, report.md
#   progress : cat results/<run name>/status.json
#   stop     : kill -TERM -- -$(cat results/<run name>/driver.pid)
#   continue : this same command

SELF=$(readlink -f "$0")
cd /home/dblab/hj/mllm_v2 || exit 2
RUN=${1:?usage: scripts/run_mt_gqa_main_v2.sh runs/mt_gqa_main_v2_<UTC>}
RUN=${RUN%/}
NAME=$(basename "$RUN")
case "$RUN" in runs/mt_gqa_main_v2_*) ;; *) echo "run must be runs/mt_gqa_main_v2_<UTC> (relative to the repository)"; exit 2;; esac
RES=results/$NAME
mkdir -p "$RES"

# ---------------------------------------------------------------- detach
if [ "${MTGQA_V2_DETACHED:-0}" != "1" ]; then
  MTGQA_V2_DETACHED=1 setsid nohup "$SELF" "$RUN" >> "$RES/driver.log" 2>&1 < /dev/null &
  echo "driver started in a new session (pid file $RES/driver.pid, log $RES/driver.log, status $RES/status.json)"
  exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export IMPRESS_OFFICIAL_REPO=/home/dblab/hj/_external_ref/HyperInfer
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
EVAL=scripts/222_eval_mt_gqa_main_v2.py; VAL=scripts/223_validate_mt_gqa_main_v2.py; REP=scripts/224_report_mt_gqa_main_v2.py
ST=scripts/llava_next_main_status.py
MAX_PHASE_STARTS=${MTGQA_V2_MAX_PHASE_STARTS:-3}      # automatic restarts of an interrupted request phase inside one driver run
GPU_WAIT_POLLS=${MTGQA_V2_GPU_WAIT_POLLS:-240}        # x 30 s: how long a GPU stage waits for another process to leave the GPU

# ------------------------------------------------------------------ lock
exec 9> "$RES/driver.lock"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) another driver holds $RES/driver.lock (pid $(cat "$RES/driver.pid" 2>/dev/null)); this one exits"
  exit 3
fi
echo $$ > "$RES/driver.pid"

log() { echo "=== $(date -u +%FT%TZ) $*"; }
status() { $PY $ST "$NAME" "$@"; }
ssd() { [ -d "$RUN" ] && $PY $ST "$RUN" --ssd-sample "$1"; }
fail() {   # fail <stage> <message>: record and stop; the report is still written
  log "STOP $1: $2"
  status stage="stopped" stopped_at_stage="$1" "last_error={\"stage\": \"$1\", \"error\": \"$2\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"
  finish 1
}
finish() {   # finish <exit code>: the report says what it can: the main tables, an interim main table, or the gate outcome
  ssd "driver_finish"
  if [ -e "$RUN/config.json" ]; then
    if [ -e "$RUN/main_execution_complete.json" ]; then
      log "START posthoc"; $PY $VAL --run "$RUN" --stage posthoc; log "END posthoc rc=$?"
    fi
    $PY $ST "$RUN" --smart after
    log "START report"; $PY $REP --run "$RUN" --phase auto; RC=$?; log "END report rc=$RC"
    [ $RC = 0 ] && status report="$RES/report.md" || status "last_error={\"stage\": \"report\", \"error\": \"report script rc=$RC (see driver.log)\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"
  fi
  if [ "$1" = 0 ]; then status stage="done" finished_utc="$(date -u +%FT%TZ)" eta_utc="$(date -u +%FT%TZ)" eta_note="finished" driver_pid=null; fi
  log "DRIVER END rc=$1"
  exit "$1"
}
on_signal() {
  log "signal received: stopping"
  status stage="stopped_by_signal" "last_error={\"stage\": \"signal\", \"error\": \"driver received TERM/INT; continue with the same command\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"
  exit 130
}
trap on_signal TERM INT

log "DRIVER START pid $$ ppid $PPID sid $(ps -o sid= -p $$ | tr -d ' ') pgid $(ps -o pgid= -p $$ | tr -d ' ') tty $(ps -o tty= -p $$ | tr -d ' ') boot $(cat /proc/sys/kernel/random/boot_id)"

# ---------------------------------------------------------------- freeze
if [ -d "$RUN" ] && [ ! -e "$RUN/protection_before.json" ]; then      # a freeze that was interrupted: nothing was measured yet
  mv "$RUN" "${RUN}.incomplete_freeze_$(date -u +%Y%m%dT%H%M%SZ)"; log "moved an incomplete freeze aside"
fi
if [ ! -d "$RUN" ]; then
  KIND=fresh
  status stage="freeze" driver_pid=$$
  log "START freeze"
  $PY $EVAL --run "$RUN" --stage freeze; RC=$?
  log "END freeze rc=$RC"
  if [ $RC != 0 ]; then status stage="stopped" "last_error={\"stage\": \"freeze\", \"error\": \"freeze rc=$RC (see driver.log)\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"; exit 1; fi
  $PY $ST "$RUN" --smart before
else
  KIND=resume
fi
$PY $ST "$RUN" --invocation $$ $KIND
rm -f "$RUN/STOP_REQUESTED" "$RUN"/NO_RETRY_*.json
ssd "driver_start_$KIND"
status stage="starting" driver_pid=$$ driver_started_utc="$(date -u +%FT%TZ)" driver_kind="$KIND" \
       driver_invocations="$(wc -l < "$RUN/driver_invocations.jsonl")" last_error=null \
       "commands={\"progress\": \"cat $RES/status.json\", \"stop\": \"kill -TERM -- -\$(cat $RES/driver.pid)\", \"continue\": \"scripts/run_mt_gqa_main_v2.sh $RUN\"}"

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
  # no result at all = the check itself broke (not a judged failure): stop without an admission record, so the same command retries it
  [ "$(gate_status "$name")" = MISSING ] && fail "$name" "the check did not produce a result (see driver.log)"
  [ "$(gate_status "$name")" = PASS ] || { status "gate_${name}=FAIL"; return 1; }
  status "gate_${name}=PASS"
}

phase() {   # phase <name>: run or continue a request phase until it is complete
  local ph=$1 n=0 rc=0
  if [ -e "$RUN/${ph}_execution_complete.json" ]; then log "SKIP ${ph} phase (complete)"; return 0; fi
  while [ $n -lt "$MAX_PHASE_STARTS" ]; do
    n=$((n + 1))
    local resume=""; [ -d "$RUN/$ph" ] && resume="--resume"
    wait_gpu_free || return 3
    status stage="${ph}_phase"
    log "START ${ph} phase $resume (start $n of at most $MAX_PHASE_STARTS in this driver run)"
    $PY $EVAL --run "$RUN" --stage "$ph" $resume; rc=$?
    log "END ${ph} phase rc=$rc"
    ssd "after_${ph}_phase"
    [ -e "$RUN/${ph}_execution_complete.json" ] && return 0
    [ -e "$RUN/STOP_REQUESTED" ] && { log "STOP_REQUESTED found"; return 2; }
    # an error that would repeat (token ids, image audit, storage, frozen files): no automatic restart
    [ -e "$RUN/NO_RETRY_${ph}.json" ] && { log "the ${ph} phase stopped on an error that a restart would repeat (runs/$NAME/NO_RETRY_${ph}.json)"; return 4; }
    sleep 60
  done
  return 1
}

# ------------------------------------------------------------ gate chain
OK=1
if [ "$(gate_status admission)" = MISSING ]; then
  check tests --stage tests || OK=0
  check tokens --stage tokens || OK=0
  if [ $OK = 1 ]; then
    if phase legacy; then
      check legacy --stage legacy || OK=0
    else fail legacy_phase "the legacy-tokenization phase did not complete (see driver.log and runs/$NAME/interruption_legacy_*.json)"; fi
    if phase gate; then
      check gate --stage gate || OK=0
    else fail gate_phase "the gate phase did not complete (see driver.log and runs/$NAME/interruption_gate_*.json)"; fi
  fi
  if [ $OK = 1 ]; then
    if phase devgate; then
      check devgate --stage devgate || OK=0
    else fail devgate_phase "the devgate phase did not complete (see driver.log and runs/$NAME/interruption_devgate_*.json)"; fi
  fi
  if [ $OK = 1 ]; then
    if phase dev; then
      check dev --stage dev || OK=0
      log "START dev pilot table"; $PY $REP --run "$RUN" --phase dev; log "END dev pilot table rc=$?"
    else fail dev_phase "the dev pilot did not complete (see driver.log and runs/$NAME/interruption_dev_*.json)"; fi
  fi
  status stage="admission"
  log "START admission"; $PY $VAL --run "$RUN" --stage admit; log "END admission rc=$? status=$(gate_status admission)"
fi
if [ "$(gate_status admission)" != PASS ]; then
  status gates="FAIL" main="not started"
  fail admission "gates not passed; the main phase is not started (details: runs/$NAME/gates/admission.json and report.md)"
fi
status gates="PASS" main="admitted"

# ------------------------------------------------------------------ main
phase main; RC=$?
if [ $RC != 0 ]; then
  fail main_phase "the main phase stopped before completion (rc=$RC; see runs/$NAME/interruption_main_*.json); continue with: scripts/run_mt_gqa_main_v2.sh $RUN"
fi
status main="complete"
finish 0

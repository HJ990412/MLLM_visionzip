#!/bin/bash
# LLaVA-NeXT x MT-VQA 25k main experiment (FullLoad / IMPRESS / Ours measured, ReComp from the v3 record):
# the whole chain, detached.  Contract: docs/llava_next_mtvqa_main_contract.md.
#
#   scripts/run_llava_next_mtvqa_main.sh runs/llava_next_mtvqa_main_<UTC>
#
# The same command starts the run and, after an interruption or a reboot, continues it.
# It re-executes itself in a new session (setsid + nohup), so it does not depend on the terminal,
# the SSH connection or the session that typed it; it prints where the PID file is and returns at once.
#
#   freeze -> pre-run checks (prerun.json) -> gate 1 (tests) -> gate 5 (scorer) -> gate phase (4 images)
#          -> gates 2, 3 -> pilot (40 images) -> gate 4 + pilot table (gate 6) -> admission
#          -> main (25,000 images) -> report
#
# Every stage decides by itself; nothing waits for a person.  A stage that already has its result is
# skipped; a request phase that was interrupted continues (--resume: committed images are skipped, the
# interrupted image starts again at turn 1).  If the pre-run checks or a gate fail, the main phase is
# not started, the failure is written to status.json and report.md, and the driver exits.
#
# Files (results/<run name>/): status.json, driver.log, driver.pid, driver.lock, prerun.json, report.md
#   progress : cat results/<run name>/status.json
#   stop     : kill -TERM -- -$(cat results/<run name>/driver.pid)
#   continue : this same command

SELF=$(readlink -f "$0")
cd /home/dblab/hj/mllm_v2 || exit 2
RUN=${1:?usage: scripts/run_llava_next_mtvqa_main.sh runs/llava_next_mtvqa_main_<UTC>}
RUN=${RUN%/}
NAME=$(basename "$RUN")
case "$RUN" in runs/llava_next_mtvqa_main_*) ;; *) echo "run must be runs/llava_next_mtvqa_main_<UTC> (relative to the repository)"; exit 2;; esac
RES=results/$NAME
mkdir -p "$RES"

# ---------------------------------------------------------------- detach
if [ "${LNEXT_MAIN_DETACHED:-0}" != "1" ]; then
  LNEXT_MAIN_DETACHED=1 setsid nohup "$SELF" "$RUN" >> "$RES/driver.log" 2>&1 < /dev/null &
  echo "driver started in a new session (pid file $RES/driver.pid, log $RES/driver.log, status $RES/status.json)"
  exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export IMPRESS_OFFICIAL_REPO=/home/dblab/hj/_external_ref/HyperInfer
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
EVAL=scripts/209_eval_llava_next_mtvqa_main.py; VAL=scripts/210_validate_llava_next_mtvqa_main.py; REP=scripts/211_report_llava_next_mtvqa_main.py
ST=scripts/llava_next_main_status.py
MAX_PHASE_STARTS=${LNEXT_MAIN_MAX_PHASE_STARTS:-5}      # automatic restarts of an interrupted request phase inside one driver run
GPU_WAIT_POLLS=${LNEXT_MAIN_GPU_WAIT_POLLS:-240}        # x 30 s: how long a GPU stage waits for another process to leave the GPU

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
finish() {   # finish <exit code>: the report says what it can: the main tables, an interim main table, or the pre-run / gate outcome
  ssd "driver_finish"
  if [ -e "$RUN/config.json" ]; then
    $PY $ST "$RUN" --smart after
    log "START report"; $PY $REP --run "$RUN" --phase auto; RC=$?; log "END report rc=$RC"
    [ $RC = 0 ] && status report="$RES/report.md" || status "last_error={\"stage\": \"report\", \"error\": \"report script rc=$RC\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"
  fi
  if [ "$1" = 0 ]; then status stage="done" finished_utc="$(date -u +%FT%TZ)" driver_pid=null; fi
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
else
  KIND=resume
fi
$PY $ST "$RUN" --invocation $$ $KIND
rm -f "$RUN/STOP_REQUESTED"
ssd "driver_start_$KIND"
status stage="starting" driver_pid=$$ driver_started_utc="$(date -u +%FT%TZ)" driver_kind="$KIND" \
       driver_invocations="$(wc -l < "$RUN/driver_invocations.jsonl")" last_error=null \
       "commands={\"progress\": \"cat $RES/status.json\", \"stop\": \"kill -TERM -- -\$(cat $RES/driver.pid)\", \"continue\": \"scripts/run_llava_next_mtvqa_main.sh $RUN\"}"

wait_gpu_free() {   # a GPU stage starts only when no other process computes on the GPU
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
  [ "$(gate_status "$name")" = PASS ]
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
    sleep 60
  done
  return 1
}

# -------------------------------------------------------- pre-run checks
if [ ! -e "$RUN/prerun_receipt.json" ] || [ "$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$RUN/prerun_receipt.json" 2>/dev/null)" != PASS ]; then
  status stage="prerun"
  log "START prerun"; $PY $EVAL --run "$RUN" --stage prerun; RC=$?; log "END prerun rc=$RC"
  $PY $ST "$RUN" --smart before
  if [ $RC != 0 ]; then
    status prerun="BLOCKED" main="not started"
    fail prerun "the pre-run checks did not pass; nothing was measured (details: $RES/prerun.json and report.md)"
  fi
else
  log "SKIP prerun (recorded: PASS)"
fi

# ------------------------------------------------------------ gate chain
OK=1
if [ "$(gate_status admission)" = MISSING ]; then
  check tests --stage tests || OK=0
  check scorer --stage scorer || OK=0
  if phase gate; then
    check gate --stage gate || OK=0
  else fail gate_phase "the gate phase did not complete (see driver.log and runs/$NAME/interruption_gate_*.json)"; fi
  if [ $OK = 1 ]; then
    if phase pilot; then
      check pilot --stage pilot || OK=0
      log "START pilot table"; $PY $REP --run "$RUN" --phase pilot; log "END pilot table rc=$?"
    else fail pilot_phase "the pilot did not complete (see driver.log and runs/$NAME/interruption_pilot_*.json)"; fi
  fi
  status stage="admission"
  log "START admission"; $PY $VAL --run "$RUN" --stage admit; log "END admission rc=$? status=$(gate_status admission)"
fi
if [ "$(gate_status admission)" != PASS ]; then
  status gates="FAIL" main="not started"
  fail admission "gates not passed; the main phase is not started (details: runs/$NAME/gates/ and report.md)"
fi
status gates="PASS" main="admitted"

# ------------------------------------------------------------------ main
phase main; RC=$?
if [ $RC != 0 ]; then
  fail main_phase "the main phase stopped before completion (rc=$RC); continue with: scripts/run_llava_next_mtvqa_main.sh $RUN"
fi
status main="complete"
finish 0

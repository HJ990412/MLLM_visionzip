#!/bin/bash
# LLaVA-1.5 x MT-GQA main experiment (ReComp / FullLoad / IMPRESS / Ours): the whole chain, detached.
# Contract: docs/llava15_mt_gqa_main_contract.md.
#
#   scripts/run_llava15_mt_gqa_main.sh runs/llava15_mt_gqa_main_<UTC>
#
# The same command starts the run and, after an interruption or a reboot, continues it.
# It re-executes itself in a new session (setsid + nohup), so it does not depend on the terminal,
# the SSH connection or the session that typed it; it prints the driver PID and returns at once.
#
#   freeze -> gate 7 (tests) -> gate phase (4 images) -> gates 1,3,4,5,6 -> gate 2 (4 images)
#          -> devgate phase (DEV 40) -> gates 3,4,5 -> gate 2 (DEV 40) -> dev pilot (DEV 40) -> gates 1,6,8
#          -> admission -> main (398 images) -> posthoc -> report
#
# Every stage decides by itself.  A stage that already has its result is skipped; a request phase that
# was interrupted continues (--resume).  If a gate fails, the remaining GPU stages and the main phase
# are not started, the failure is written to status.json and report.md, and the driver exits.
#
# Files (results/<run name>/): status.json, driver.log, driver.pid, driver.lock, report.md
#   progress : cat results/<run name>/status.json
#   stop     : kill -TERM -- -$(cat results/<run name>/driver.pid)
#   continue : this same command

SELF=$(readlink -f "$0")
cd /home/dblab/hj/mllm_v2 || exit 2
RUN=${1:?usage: scripts/run_llava15_mt_gqa_main.sh runs/llava15_mt_gqa_main_<UTC>}
RUN=${RUN%/}
NAME=$(basename "$RUN")
case "$RUN" in runs/llava15_mt_gqa_main_*) ;; *) echo "run must be runs/llava15_mt_gqa_main_<UTC> (relative to the repository)"; exit 2;; esac
RES=results/$NAME
mkdir -p "$RES"

# ---------------------------------------------------------------- detach
if [ "${L15_MAIN_DETACHED:-0}" != "1" ]; then
  L15_MAIN_DETACHED=1 setsid nohup "$SELF" "$RUN" >> "$RES/driver.log" 2>&1 < /dev/null &
  echo "driver started in a new session (pid file $RES/driver.pid, log $RES/driver.log, status $RES/status.json)"
  exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export IMPRESS_OFFICIAL_REPO=/home/dblab/hj/_external_ref/HyperInfer
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
EVAL=scripts/206_eval_llava15_mt_gqa_main.py; VAL=scripts/207_validate_llava15_mt_gqa_main.py; REP=scripts/208_report_llava15_mt_gqa_main.py
ST=scripts/llava15_main_status.py
MAX_PHASE_STARTS=${L15_MAIN_MAX_PHASE_STARTS:-3}      # automatic restarts of an interrupted request phase inside one driver run
GPU_WAIT_POLLS=${L15_MAIN_GPU_WAIT_POLLS:-240}        # x 30 s: how long a GPU stage waits for another process to leave the GPU

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
finish() {   # finish <exit code>: report (main when the main phase is complete, else the gate / interruption report)
  ssd "driver_finish"
  if [ -e "$RUN/config.json" ]; then
    if [ -e "$RUN/main_execution_complete.json" ]; then
      log "START posthoc"; $PY $VAL --run "$RUN" --stage posthoc; log "END posthoc rc=$?"
    fi
    # the report script decides what it can report: the main tables, an interim main table, or the gate outcome only
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
  log "START freeze"
  SB=""; [ -n "${L15_MAIN_SOURCE_BEFORE:-}" ] && SB="--source-before $L15_MAIN_SOURCE_BEFORE"
  $PY $EVAL --run "$RUN" --stage freeze $SB; RC=$?
  log "END freeze rc=$RC"
  if [ $RC != 0 ]; then status stage="stopped" "last_error={\"stage\": \"freeze\", \"error\": \"freeze rc=$RC (see driver.log)\", \"at_utc\": \"$(date -u +%FT%TZ)\"}"; exit 1; fi
  $PY - "$RUN" "$RES" <<'EOF'
import json, sys, time, pathlib
run, res = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
plan = json.loads((run / "storage_plan.json").read_text())
doc = {"written_before_any_gpu_measurement_unix": time.time(), "free_bytes_at_freeze": plan["free_bytes_at_freeze"],
       "free_gib_at_freeze": plan["free_bytes_at_freeze"] / 2**30, "required_free_bytes": plan["required_free_bytes"],
       "expected_ssd_write_bytes": plan["expected_ssd_write_bytes"], "expected_ssd_write_note": plan["expected_ssd_write_note"],
       "per_image_bundle_bytes": plan["per_image_bundle_bytes"]}
(res / "prerun.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
EOF
else
  KIND=resume
fi
$PY $ST "$RUN" --invocation $$ $KIND
rm -f "$RUN/STOP_REQUESTED"
ssd "driver_start_$KIND"
status stage="starting" driver_pid=$$ driver_started_utc="$(date -u +%FT%TZ)" driver_kind="$KIND" \
       driver_invocations="$(wc -l < "$RUN/driver_invocations.jsonl")" last_error=null \
       "commands={\"progress\": \"cat $RES/status.json\", \"stop\": \"kill -TERM -- -\$(cat $RES/driver.pid)\", \"continue\": \"scripts/run_llava15_mt_gqa_main.sh $RUN\"}"

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

# ------------------------------------------------------------ gate chain
OK=1
if [ "$(gate_status admission)" = MISSING ]; then
  check tests --stage tests || OK=0
  if phase gate; then
    check gate --stage gate || OK=0
    check qindep_gate --stage qindep --phase gate || OK=0
  else fail gate_phase "the gate phase did not complete (see driver.log and runs/$NAME/interruption_gate_*.json)"; fi
  if [ $OK = 1 ]; then
    if phase devgate; then
      check devgate --stage devgate || OK=0
      check qindep_devgate --stage qindep --phase devgate || OK=0
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
status gates="PASS"

# ------------------------------------------------------------------ main
phase main; RC=$?
if [ $RC != 0 ]; then
  fail main_phase "the main phase stopped before completion (rc=$RC); continue with: scripts/run_llava15_mt_gqa_main.sh $RUN"
fi
finish 0

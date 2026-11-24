#!/bin/bash
# ConvBench baseline alignment (ReComp vs MetaCompress base model): stages 2-5 and diagnostics, detached.
# Plan and rules: results/convbench_alignment_<UTC>/notes.md.  Orchestrator: scripts/215_convbench_alignment_driver.py.
#
#   scripts/run_convbench_alignment.sh results/convbench_alignment_<UTC>
#
# The same command starts the run and, after an interruption or a reboot, continues it (finished
# conversations, judgments and steps are skipped).  It re-executes itself in a new session
# (setsid + nohup), so it does not depend on the terminal, the SSH connection or the session that
# typed it; it prints where to look and returns at once.  There is no point that waits for input.
#
# Files (results/<run>/): status.json, driver.log, driver.pid, driver.lock, report.md, logs/<step>.log
#   progress : cat results/<run>/status.json
#   stop     : kill -TERM -- -$(cat results/<run>/driver.pid)
#   continue : this same command

SELF=$(readlink -f "$0")
cd /home/dblab/hj/mllm_v2 || exit 2
RES=${1:?usage: scripts/run_convbench_alignment.sh results/convbench_alignment_<UTC>}
RES=${RES%/}
case "$RES" in results/convbench_alignment_*) ;; *) echo "run must be results/convbench_alignment_<UTC> (relative to the repository)"; exit 2;; esac
[ -f "$RES/notes.md" ] || { echo "$RES/notes.md missing: stage 1 and the plan must exist first"; exit 2; }

if [ "${CONVBENCH_ALIGN_DETACHED:-0}" != "1" ]; then
  CONVBENCH_ALIGN_DETACHED=1 setsid nohup "$SELF" "$RES" >> "$RES/driver.log" 2>&1 < /dev/null &
  echo "driver started in a new session (pid file $RES/driver.pid, log $RES/driver.log, status $RES/status.json)"
  exit 0
fi

exec 9> "$RES/driver.lock"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) another driver holds $RES/driver.lock (pid $(cat "$RES/driver.pid" 2>/dev/null)); this one exits"
  exit 3
fi
echo $$ > "$RES/driver.pid"
echo "=== $(date -u +%FT%TZ) DRIVER START pid $$ ppid $PPID sid $(ps -o sid= -p $$ | tr -d ' ') pgid $(ps -o pgid= -p $$ | tr -d ' ') tty $(ps -o tty= -p $$ | tr -d ' ') boot $(cat /proc/sys/kernel/random/boot_id)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.
# exec: the orchestrator keeps this PID (= process group and session leader) and the lock on fd 9.
exec /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/215_convbench_alignment_driver.py --res "$RES"

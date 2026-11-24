#!/bin/bash
# ConvBench main experiment (LLaVA-NeXT, LLaVA-1.5 x ReComp, FullLoad, IMPRESS, Ours): the whole chain, detached.
# Contract: docs/convbench_main_contract.md.  Orchestrator: scripts/221_convbench_main_driver.py.
#
#   scripts/run_convbench_main.sh results/convbench_main_<UTC>
#
# The same command starts the run and, after an interruption or a reboot, continues it: finished steps,
# committed conversations and finished judgments are skipped; a conversation that was interrupted starts again
# at turn 1 for all four methods.  It re-executes itself in a new session (setsid + nohup), so it does not
# depend on the terminal, the SSH connection or the session that typed it; it prints where to look and
# returns at once.  There is no point that waits for input.
#
# Files (results/<run>/): status.json, driver.log, driver.pid, driver.lock, prerun.json, report.md
#   progress : cat results/<run>/status.json
#   stop     : kill -TERM -- -$(cat results/<run>/driver.pid)
#   continue : this same command

SELF=$(readlink -f "$0")
cd /home/dblab/hj/mllm_v2 || exit 2
RES=${1:?usage: scripts/run_convbench_main.sh results/convbench_main_<UTC>}
RES=${RES%/}
case "$RES" in results/convbench_main_*) ;; *) echo "run must be results/convbench_main_<UTC> (relative to the repository)"; exit 2;; esac
[ -f docs/convbench_main_contract.md ] || { echo "docs/convbench_main_contract.md missing: the contract must exist before the run"; exit 2; }
mkdir -p "$RES"

if [ "${CONVBENCH_MAIN_DETACHED:-0}" != "1" ]; then
  CONVBENCH_MAIN_DETACHED=1 setsid nohup "$SELF" "$RES" >> "$RES/driver.log" 2>&1 < /dev/null &
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
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# exec: the orchestrator keeps this PID (= process group and session leader) and the lock on fd 9.
exec /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/221_convbench_main_driver.py --res "$RES"

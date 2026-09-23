#!/usr/bin/env bash
# Launch/resume the MT-VQA-v2-reconstructed Generated-History experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="/home/dblab/anaconda3/envs/mllm_ft/bin/python"
PREFIX="mt_vqa_v2_generated_4arm_"
ORCHESTRATOR="${ROOT}/scripts/62_run_mt_vqa_v2_generated.py"

usage() {
  printf '%s\n' \
    "Usage:" \
    "  $0 [--session NAME] [--stop-after smoke|full]" \
    "  $0 --resume RUN_DIR [--session NAME] [--stop-after smoke|full]" \
    "" \
    "A fresh launch runs Generated-History smoke validation before full." \
    "Experiment roots are created by artifact protection, not this launcher."
}

resume_dir=""
session=""
stop_after="full"
while (($#)); do
  case "$1" in
    --resume)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      resume_dir="$2"; shift 2 ;;
    --session)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      session="$2"; shift 2 ;;
    --stop-after)
      [[ $# -ge 2 && ( "$2" == "smoke" || "$2" == "full" ) ]] || {
        printf '%s\n' '--stop-after must be smoke or full' >&2; exit 2;
      }
      stop_after="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2 ;;
  esac
done

command -v tmux >/dev/null || { printf '%s\n' 'tmux is required' >&2; exit 1; }
command -v flock >/dev/null || { printf '%s\n' 'flock is required' >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; }
[[ -f "$ORCHESTRATOR" ]] || { printf 'Missing orchestrator: %s\n' "$ORCHESTRATOR" >&2; exit 1; }

# This lock is outside runs/results, so launcher serialization does not change
# the protection snapshot that must precede either experiment-root creation.
exec 9>"/tmp/mllm_v2_mt_vqa_v2_generated_launcher.lock"
flock -n 9 || {
  printf '%s\n' 'Another MT-VQA-v2 Generated-History launcher is initializing' >&2
  exit 1
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "$resume_dir" ]]; then
  run_dir="$(realpath "$resume_dir")"
  run_name="$(basename "$run_dir")"
  [[ "$(dirname "$run_dir")" == "${ROOT}/runs" ]] || {
    printf 'Run directory is outside this project: %s\n' "$run_dir" >&2; exit 1;
  }
  [[ "$run_name" == "${PREFIX}"* ]] || {
    printf 'Not an MT-VQA-v2 Generated-History run: %s\n' "$run_dir" >&2; exit 1;
  }
  [[ -d "$run_dir" && -f "$run_dir/manifest.json" ]] || {
    printf 'Not a resumable run directory: %s\n' "$run_dir" >&2; exit 1;
  }
  results_dir="${ROOT}/results/${run_name}"
  [[ ! -f "$run_dir/COMPLETED" ]] || {
    printf 'Run is already complete: %s\n' "$run_dir" >&2; exit 1;
  }
  [[ -n "$session" ]] || session="mtvqa_v2_generated_resume_${timestamp}"
  resume_flag=(--resume)
else
  run_name="${PREFIX}${timestamp}"
  run_dir="${ROOT}/runs/${run_name}"
  results_dir="${ROOT}/results/${run_name}"
  # Do not create either root here. The orchestrator first snapshots every
  # pre-existing run/result artifact, then the protector creates the run root.
  [[ ! -e "$run_dir" && ! -e "$results_dir" ]] || {
    printf 'Fresh experiment root already exists: %s\n' "$run_name" >&2; exit 1;
  }
  [[ -n "$session" ]] || session="mtvqa_v2_generated_${timestamp}"
  resume_flag=()
fi

[[ "$session" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  printf 'Unsafe tmux session name: %s\n' "$session" >&2; exit 2;
}
if tmux has-session -t "=$session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session" >&2; exit 1
fi
if tmux list-sessions -F '#S' 2>/dev/null \
    | grep -Eq '^mtvqa_v2_generated(_resume)?_[A-Za-z0-9_.-]+$'; then
  printf '%s\n' 'An MT-VQA-v2 Generated-History run already has a live tmux session' >&2
  exit 1
fi

command_file="/tmp/${run_name}_${timestamp}_command.sh"
bootstrap_log="/tmp/${run_name}_${timestamp}_launcher.log"
orchestrator=(
  "$PYTHON_BIN" "$ORCHESTRATOR"
  --run-root "$run_dir"
  --results-root "$results_dir"
  --stop-after "$stop_after"
  "${resume_flag[@]}"
)

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'export HF_HUB_OFFLINE=1\n'
  printf 'export HF_DATASETS_OFFLINE=1\n'
  printf 'export TRANSFORMERS_OFFLINE=1\n'
  printf 'export TOKENIZERS_PARALLELISM=false\n'
  printf 'cd %q\n' "$ROOT"
  printf 'exec'
  printf ' %q' "${orchestrator[@]}"
  printf '\n'
} > "$command_file"
chmod 0700 "$command_file"

tmux new-session -d -s "$session" \
  "bash $(printf '%q' "$command_file") >> $(printf '%q' "$bootstrap_log") 2>&1"

printf 'Started detached tmux session: %s\n' "$session"
printf 'Run directory: %s\n' "$run_dir"
printf 'Results directory: %s\n' "$results_dir"
printf 'Bootstrap log (before protected root exists): %s\n' "$bootstrap_log"
printf 'Follow: tail -f %q\n' "$bootstrap_log"
printf 'Attach: tmux attach -t %q\n' "$session"

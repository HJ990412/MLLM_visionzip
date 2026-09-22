#!/usr/bin/env bash
# Launch or resume QA-Chunk25 in a detached, persistent tmux session.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="/home/dblab/anaconda3/envs/mllm_ft/bin/python"
RUN_ROOT="${ROOT}/runs/query_aware_chunk_baseline"
RESULTS_ROOT="${ROOT}/results/query_aware_chunk_baseline"
STORE_DIR="${ROOT}/runs/query_aware_baseline/gqa40_240_final_store"
REFERENCE_RUN_DIR="${ROOT}/runs/query_aware_baseline/gqa40_240_final"

usage() {
  printf '%s\n' \
    "Usage:" \
    "  $0 --mode smoke|full [--session NAME] [--max-new-requests N]" \
    "  $0 --resume RUN_DIR [--session NAME] [--max-new-requests N]" \
    "" \
    "The new-run form creates timestamped run/results directories." \
    "The resume form continues durable request IDs from results_partial.jsonl."
}

mode=""
resume_dir=""
session=""
max_new_requests=""
while (($#)); do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      mode="$2"; shift 2 ;;
    --resume)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      resume_dir="$2"; shift 2 ;;
    --session)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      session="$2"; shift 2 ;;
    --max-new-requests)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || {
        printf '%s\n' '--max-new-requests requires a positive integer' >&2
        exit 2
      }
      max_new_requests="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -n "$mode" && -n "$resume_dir" ]] || [[ -z "$mode" && -z "$resume_dir" ]]; then
  usage >&2
  exit 2
fi
command -v tmux >/dev/null || { printf 'tmux is required\n' >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; }

mkdir -p "$RUN_ROOT" "$RESULTS_ROOT"
exec 9>"$RUN_ROOT/.launcher.lock"
flock -n 9 || {
  printf 'Another QA-Chunk launcher is initializing a run\n' >&2
  exit 1
}
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -n "$resume_dir" ]]; then
  run_dir="$(realpath "$resume_dir")"
  [[ -d "$run_dir" && -f "$run_dir/manifest.json" ]] || {
    printf 'Not a resumable run directory: %s\n' "$run_dir" >&2
    exit 1
  }
  [[ ! -e "$run_dir/COMPLETED" ]] || {
    printf 'Run is already complete: %s\n' "$run_dir" >&2
    exit 1
  }
  if [[ -f "$run_dir/tmux_session.txt" ]]; then
    while IFS= read -r prior_session; do
      if [[ -n "$prior_session" ]] && tmux has-session -t "=$prior_session" 2>/dev/null; then
        printf 'Run already has a live tmux session: %s\n' "$prior_session" >&2
        exit 1
      fi
    done < <(sed -n 's/.*session=\([^[:space:]]*\).*/\1/p' "$run_dir/tmux_session.txt")
  fi
  run_name="$(basename "$run_dir")"
  [[ -n "$session" ]] || session="qa_chunk25_resume_${timestamp}"
  command_file="$run_dir/resume_command_${timestamp}.sh"
  evaluator=("$PYTHON_BIN" "$ROOT/scripts/52_eval_query_aware_chunk_baseline.py"
             --resume "$run_dir")
else
  case "$mode" in
    smoke) max_images=5; stem="smoke_5" ;;
    full) max_images=40; stem="gqa40_240" ;;
    *) printf 'Invalid --mode: %s\n' "$mode" >&2; usage >&2; exit 2 ;;
  esac
  if tmux list-sessions -F '#S' 2>/dev/null | grep -Eq "^qa_chunk25_${stem}_[A-Za-z0-9_.-]+$"; then
    printf 'A %s QA-Chunk run already has a live tmux session\n' "$stem" >&2
    exit 1
  fi
  run_name="${stem}_${timestamp}"
  run_dir="$RUN_ROOT/$run_name"
  results_dir="$RESULTS_ROOT/$run_name"
  mkdir "$run_dir"
  [[ ! -e "$results_dir" ]] || {
    printf 'Results directory already exists: %s\n' "$results_dir" >&2
    exit 1
  }
  [[ -n "$session" ]] || session="qa_chunk25_${stem}_${timestamp}"
  command_file="$run_dir/command.sh"
  evaluator=("$PYTHON_BIN" "$ROOT/scripts/52_eval_query_aware_chunk_baseline.py"
             --run-dir "$run_dir"
             --results-dir "$results_dir"
             --index "$ROOT/data/index.json"
             --store-dir "$STORE_DIR"
             --reference-run-dir "$REFERENCE_RUN_DIR"
             --max-images "$max_images"
             --skip 4 --questions 6 --seed 1234 --max-new-tokens 16
             --expected-index-sha256
             514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a
             --expected-workload-sha256
             97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66
             --expected-images 40 --expected-questions 240)
fi

if [[ -n "$max_new_requests" ]]; then
  evaluator+=(--max-new-requests "$max_new_requests")
fi

[[ "$session" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  printf 'Unsafe tmux session name: %s\n' "$session" >&2
  exit 2
}
if tmux has-session -t "=$session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session" >&2
  exit 1
fi

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'export HF_HUB_OFFLINE=1\n'
  printf 'export HF_DATASETS_OFFLINE=1\n'
  printf 'export TRANSFORMERS_OFFLINE=1\n'
  printf 'export TOKENIZERS_PARALLELISM=false\n'
  printf 'cd %q\n' "$ROOT"
  printf 'exec'
  printf ' %q' "${evaluator[@]}"
  printf '\n'
} > "$command_file"
chmod 0755 "$command_file"

if [[ ! -e "$run_dir/environment.txt" ]]; then
  {
    printf 'captured_at_utc=%s\n' "$timestamp"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'kernel=%s\n' "$(uname -srmo)"
    "$PYTHON_BIN" --version 2>&1
    "$PYTHON_BIN" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda))'
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
        || printf 'nvidia_smi_unavailable\n'
    fi
    printf 'offline_hf=1\noffline_datasets=1\noffline_transformers=1\n'
  } > "$run_dir/environment.txt"
fi

if [[ ! -e "$run_dir/git_state.txt" ]]; then
  {
    printf 'captured_at_utc=%s\n' "$timestamp"
    if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git -C "$ROOT" rev-parse HEAD
      git -C "$ROOT" status --short
    else
      printf 'not_a_git_worktree\n'
    fi
    sha256sum \
      "$ROOT/mmimpress/sparsevlm.py" \
      "$ROOT/mmimpress/serve.py" \
      "$ROOT/mmimpress/store.py" \
      "$ROOT/scripts/49_eval_query_aware_baseline.py" \
      "$ROOT/scripts/52_eval_query_aware_chunk_baseline.py"
  } > "$run_dir/git_state.txt"
fi

touch "$run_dir/run.log"
printf '%s\tsession=%s\tcommand=%s\n' "$timestamp" "$session" "$command_file" \
  >> "$run_dir/tmux_session.txt"

# The evaluator itself also owns a non-blocking fcntl lock, so two launchers
# racing past this point still cannot append duplicate request rows.
tmux new-session -d -s "$session" \
  "bash $(printf '%q' "$command_file") >> $(printf '%q' "$run_dir/run.log") 2>&1"

printf 'Started detached tmux session: %s\n' "$session"
printf 'Run directory: %s\n' "$run_dir"
printf 'Log: %s\n' "$run_dir/run.log"
printf 'Progress: %s\n' "$run_dir/progress.json"
printf 'Attach: tmux attach -t %q\n' "$session"
printf 'Follow: tail -f %q\n' "$run_dir/run.log"

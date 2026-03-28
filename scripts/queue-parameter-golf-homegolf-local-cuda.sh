#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
runner_path="${script_dir}/run-parameter-golf-homegolf-local-cuda.sh"

wait_for_run_root=""
poll_seconds="60"
sync_main="1"
wait_for_pid_file="0"
runner_args=()

usage() {
  cat <<'EOF' >&2
Usage: scripts/queue-parameter-golf-homegolf-local-cuda.sh [queue-options] -- [runner-options]

Queue options:
  --wait-for-run-root <path>            Existing HOMEGOLF output root whose trainer pid gates launch.
  --poll-seconds <n>                    Poll interval while waiting. Default: 60
  --wait-for-pid-file                   Wait until <run-root>/train.pid exists before waiting on its process.
  --skip-sync-main                      Do not add --sync-main when launching the runner.
  --help|-h                             Show this help text.

Everything after `--` is passed through to:
  scripts/run-parameter-golf-homegolf-local-cuda.sh

Example:
  scripts/queue-parameter-golf-homegolf-local-cuda.sh \
    --wait-for-run-root ~/scratch/psionic_homegolf_runs/current_run \
    -- \
    --run-id next_run \
    --challenge-max-steps 2 \
    --grad-accum-steps 64 \
    --grad-clip-norm 1.0 \
    --learning-rate-scale 0.75 \
    --final-validation-mode artifact_only \
    --background \
    --attach-prompt-closeout \
    --attach-detached-score-closeout
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-for-run-root)
      wait_for_run_root="$2"
      shift 2
      ;;
    --poll-seconds)
      poll_seconds="$2"
      shift 2
      ;;
    --wait-for-pid-file)
      wait_for_pid_file="1"
      shift
      ;;
    --skip-sync-main)
      sync_main="0"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      runner_args=("$@")
      break
      ;;
    *)
      echo "error: unknown queue argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${wait_for_run_root}" ]]; then
  echo "error: --wait-for-run-root is required" >&2
  exit 1
fi

if [[ ! -x "${runner_path}" ]]; then
  echo "error: runner ${runner_path} is not executable" >&2
  exit 1
fi

if [[ ${#runner_args[@]} -eq 0 ]]; then
  echo "error: missing runner arguments after --" >&2
  exit 1
fi

train_pid_path="${wait_for_run_root}/train.pid"

echo "queue_wait_root=${wait_for_run_root}"
echo "queue_poll_seconds=${poll_seconds}"
echo "queue_wait_for_pid_file=${wait_for_pid_file}"

if [[ "${wait_for_pid_file}" == "1" ]]; then
  while [[ ! -f "${train_pid_path}" ]]; do
    echo "queue_waiting_for_pid_file timestamp=$(date -Is) path=${train_pid_path}"
    sleep "${poll_seconds}"
  done
fi

if [[ -f "${train_pid_path}" ]]; then
  train_pid="$(cat "${train_pid_path}")"
  echo "queue_wait_pid=${train_pid}"
  while kill -0 "${train_pid}" 2>/dev/null; do
    echo "queue_waiting timestamp=$(date -Is) pid=${train_pid}"
    sleep "${poll_seconds}"
  done
fi

echo "queue_launch timestamp=$(date -Is)"

launch_args=()
if [[ "${sync_main}" == "1" ]]; then
  launch_args+=(--sync-main)
fi
launch_args+=("${runner_args[@]}")

cd "${repo_root}"
exec "${runner_path}" "${launch_args[@]}"

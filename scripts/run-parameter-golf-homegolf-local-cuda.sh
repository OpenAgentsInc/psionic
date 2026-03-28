#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

run_id=""
output_root=""
dataset_root="${HOME}/code/parameter-golf/data/datasets/fineweb10B_sp1024"
tokenizer_path="${HOME}/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
tmpdir="${HOME}/scratch/tmp"
cargo_target_dir="${repo_root}/target"
binary_path=""
max_steps=""
challenge_max_steps=""
grad_clip_norm=""
learning_rate_scale=""
ema_decay=""
final_model_surface=""
final_validation_mode="roundtrip_only"
validation_eval_mode="non_overlapping"
background="0"
sync_main="0"
allow_dirty="0"
attach_prompt_closeout="0"
attach_detached_score_closeout="0"
closeout_prompt="the meaning of life is"
closeout_max_new_tokens="32"
closeout_poll_seconds="30"
closeout_timeout_seconds="0"
closeout_binary_path=""
score_poll_seconds="30"
score_timeout_seconds="0"
score_binary_path=""
trainer_args=()

usage() {
  cat <<'EOF' >&2
Usage: scripts/run-parameter-golf-homegolf-local-cuda.sh [options]

Options:
  --run-id <id>                         Stable run identifier.
  --output-root <path>                  Output root. Default: ~/scratch/psionic_homegolf_runs/<run_id>
  --dataset-root <path>                 FineWeb dataset root.
  --tokenizer-path <path>               Tokenizer path.
  --tmpdir <path>                       Scratch TMPDIR for rustc and trainer temp files.
  --cargo-target-dir <path>             Cargo target dir when running through cargo.
  --binary-path <path>                  Use an existing trainer binary instead of cargo.
  --max-steps <n>                       Optional bounded max-steps override.
  --challenge-max-steps <n>             Honest challenge max-steps cap that preserves the local 600s contract.
  --grad-clip-norm <f32>                Optional training grad-clip norm override for the HOMEGOLF lane.
  --learning-rate-scale <f32>           Optional uniform learning-rate scale across HOMEGOLF optimizer groups.
  --ema-decay <f32>                     Optional EMA decay override for HOMEGOLF final-model export.
  --final-model-surface <surface>       Optional final export surface. Example: raw, ema, swa
  --final-validation-mode <mode>        Final validation mode. Default: roundtrip_only
  --validation-eval-mode <mode>         Validation eval mode. Default: non_overlapping
  --trainer-arg <arg>                   Extra trailing trainer arg. Repeatable.
  --background                          Launch in background and write a pid file.
  --attach-prompt-closeout              Launch the prompt closeout watcher beside the trainer.
  --attach-detached-score-closeout      Launch the detached artifact score watcher beside the trainer.
  --closeout-prompt <text>              Prompt used by the closeout watcher.
  --closeout-max-new-tokens <n>         Max generated tokens for closeout. Default: 32
  --closeout-poll-seconds <n>           Poll interval for closeout watcher. Default: 30
  --closeout-timeout-seconds <n>        Optional closeout timeout. Default: 0
  --closeout-binary-path <path>         Optional prompt binary for closeout watcher.
  --score-poll-seconds <n>              Poll interval for detached score watcher. Default: 30
  --score-timeout-seconds <n>           Optional detached score timeout. Default: 0
  --score-binary-path <path>            Optional detached score binary for the score watcher.
  --sync-main                           Run git pull --ff-only before launch.
  --allow-dirty                         Skip the clean-worktree refusal.
  --help|-h                             Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --dataset-root)
      dataset_root="$2"
      shift 2
      ;;
    --tokenizer-path)
      tokenizer_path="$2"
      shift 2
      ;;
    --tmpdir)
      tmpdir="$2"
      shift 2
      ;;
    --cargo-target-dir)
      cargo_target_dir="$2"
      shift 2
      ;;
    --binary-path)
      binary_path="$2"
      shift 2
      ;;
    --max-steps)
      max_steps="$2"
      shift 2
      ;;
    --challenge-max-steps)
      challenge_max_steps="$2"
      shift 2
      ;;
    --grad-clip-norm)
      grad_clip_norm="$2"
      shift 2
      ;;
    --learning-rate-scale)
      learning_rate_scale="$2"
      shift 2
      ;;
    --ema-decay)
      ema_decay="$2"
      shift 2
      ;;
    --final-model-surface)
      final_model_surface="$2"
      shift 2
      ;;
    --final-validation-mode)
      final_validation_mode="$2"
      shift 2
      ;;
    --validation-eval-mode)
      validation_eval_mode="$2"
      shift 2
      ;;
    --trainer-arg)
      trainer_args+=("$2")
      shift 2
      ;;
    --background)
      background="1"
      shift
      ;;
    --attach-prompt-closeout)
      attach_prompt_closeout="1"
      shift
      ;;
    --attach-detached-score-closeout)
      attach_detached_score_closeout="1"
      shift
      ;;
    --closeout-prompt)
      closeout_prompt="$2"
      shift 2
      ;;
    --closeout-max-new-tokens)
      closeout_max_new_tokens="$2"
      shift 2
      ;;
    --closeout-poll-seconds)
      closeout_poll_seconds="$2"
      shift 2
      ;;
    --closeout-timeout-seconds)
      closeout_timeout_seconds="$2"
      shift 2
      ;;
    --closeout-binary-path)
      closeout_binary_path="$2"
      shift 2
      ;;
    --score-poll-seconds)
      score_poll_seconds="$2"
      shift 2
      ;;
    --score-timeout-seconds)
      score_timeout_seconds="$2"
      shift 2
      ;;
    --score-binary-path)
      score_binary_path="$2"
      shift 2
      ;;
    --sync-main)
      sync_main="1"
      shift
      ;;
    --allow-dirty)
      allow_dirty="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${run_id}" ]]; then
  run_id="homegolf-local-cuda-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "${output_root}" ]]; then
  output_root="${HOME}/scratch/psionic_homegolf_runs/${run_id}"
fi

if [[ -n "${max_steps}" ]] && [[ -n "${challenge_max_steps}" ]]; then
  echo "error: --max-steps and --challenge-max-steps are mutually exclusive" >&2
  exit 1
fi

if [[ "${attach_prompt_closeout}" == "1" ]] && [[ "${background}" != "1" ]]; then
  echo "error: --attach-prompt-closeout requires --background" >&2
  exit 1
fi
if [[ "${attach_detached_score_closeout}" == "1" ]] && [[ "${background}" != "1" ]]; then
  echo "error: --attach-detached-score-closeout requires --background" >&2
  exit 1
fi

mkdir -p "${output_root}" "${tmpdir}"

status_branch="$(git -C "${repo_root}" status --short --branch)"
current_branch="$(git -C "${repo_root}" rev-parse --abbrev-ref HEAD)"
if [[ "${current_branch}" != "main" ]]; then
  echo "error: HOMEGOLF local CUDA runner requires branch main, found ${current_branch}" >&2
  exit 1
fi
if [[ "${allow_dirty}" != "1" ]] && [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "error: HOMEGOLF local CUDA runner requires a clean worktree" >&2
  echo "${status_branch}" >&2
  exit 1
fi

if [[ "${sync_main}" == "1" ]]; then
  git -C "${repo_root}" pull --ff-only
  status_branch="$(git -C "${repo_root}" status --short --branch)"
fi

if [[ ! -d "${dataset_root}" ]]; then
  echo "error: dataset root ${dataset_root} does not exist" >&2
  exit 1
fi
if [[ ! -f "${tokenizer_path}" ]]; then
  echo "error: tokenizer path ${tokenizer_path} does not exist" >&2
  exit 1
fi

report_path="${output_root}/training_report.json"
log_path="${output_root}/train.log"
pid_path="${output_root}/train.pid"
launch_env_path="${output_root}/launch.env"
prompt_report_path="${output_root}/prompt_report.json"
closeout_summary_path="${output_root}/closeout_summary.json"
prompt_log_path="${output_root}/prompt_closeout.log"
prompt_pid_path="${output_root}/prompt_closeout.pid"
score_report_path="${output_root}/artifact_score_report.json"
score_summary_path="${output_root}/artifact_score_summary.json"
score_log_path="${output_root}/artifact_score_closeout.log"
score_pid_path="${output_root}/artifact_score_closeout.pid"

trainer_command=()
executor_label="cargo"
if [[ -n "${binary_path}" ]]; then
  if [[ ! -x "${binary_path}" ]]; then
    echo "error: binary path ${binary_path} is not executable" >&2
    exit 1
  fi
  executor_label="binary"
  trainer_command=("${binary_path}")
else
  trainer_command=(
    cargo
    run
    -q
    -p
    psionic-train
    --bin
    parameter_golf_homegolf_single_cuda_train
    --
  )
fi

trainer_command+=("${dataset_root}" "${tokenizer_path}" "${report_path}")
if [[ -n "${max_steps}" ]]; then
  trainer_command+=("${max_steps}")
fi
trainer_command+=("${final_validation_mode}" "${validation_eval_mode}")
if [[ ${#trainer_args[@]} -gt 0 ]]; then
  trainer_command+=("${trainer_args[@]}")
fi

printf 'RUN_ID=%q\n' "${run_id}" > "${launch_env_path}"
printf 'REPO_ROOT=%q\n' "${repo_root}" >> "${launch_env_path}"
printf 'STATUS_BRANCH=%q\n' "${status_branch}" >> "${launch_env_path}"
printf 'TMPDIR=%q\n' "${tmpdir}" >> "${launch_env_path}"
printf 'CARGO_TARGET_DIR=%q\n' "${cargo_target_dir}" >> "${launch_env_path}"
printf 'OUTPUT_ROOT=%q\n' "${output_root}" >> "${launch_env_path}"
printf 'REPORT_PATH=%q\n' "${report_path}" >> "${launch_env_path}"
printf 'LOG_PATH=%q\n' "${log_path}" >> "${launch_env_path}"
printf 'EXECUTOR=%q\n' "${executor_label}" >> "${launch_env_path}"
printf 'DATASET_ROOT=%q\n' "${dataset_root}" >> "${launch_env_path}"
printf 'TOKENIZER_PATH=%q\n' "${tokenizer_path}" >> "${launch_env_path}"
printf 'FINAL_VALIDATION_MODE=%q\n' "${final_validation_mode}" >> "${launch_env_path}"
printf 'VALIDATION_EVAL_MODE=%q\n' "${validation_eval_mode}" >> "${launch_env_path}"
if [[ -n "${max_steps}" ]]; then
  printf 'MAX_STEPS=%q\n' "${max_steps}" >> "${launch_env_path}"
fi
if [[ -n "${challenge_max_steps}" ]]; then
  printf 'CHALLENGE_MAX_STEPS=%q\n' "${challenge_max_steps}" >> "${launch_env_path}"
fi
if [[ -n "${grad_clip_norm}" ]]; then
  printf 'GRAD_CLIP_NORM=%q\n' "${grad_clip_norm}" >> "${launch_env_path}"
fi
if [[ -n "${learning_rate_scale}" ]]; then
  printf 'LEARNING_RATE_SCALE=%q\n' "${learning_rate_scale}" >> "${launch_env_path}"
fi
if [[ -n "${ema_decay}" ]]; then
  printf 'EMA_DECAY=%q\n' "${ema_decay}" >> "${launch_env_path}"
fi
if [[ -n "${final_model_surface}" ]]; then
  printf 'FINAL_MODEL_SURFACE=%q\n' "${final_model_surface}" >> "${launch_env_path}"
fi
if [[ -n "${binary_path}" ]]; then
  printf 'BINARY_PATH=%q\n' "${binary_path}" >> "${launch_env_path}"
fi
if [[ "${attach_prompt_closeout}" == "1" ]]; then
  printf 'PROMPT_REPORT_PATH=%q\n' "${prompt_report_path}" >> "${launch_env_path}"
  printf 'CLOSEOUT_SUMMARY_PATH=%q\n' "${closeout_summary_path}" >> "${launch_env_path}"
  printf 'PROMPT_LOG_PATH=%q\n' "${prompt_log_path}" >> "${launch_env_path}"
  printf 'CLOSEOUT_PROMPT=%q\n' "${closeout_prompt}" >> "${launch_env_path}"
  printf 'CLOSEOUT_MAX_NEW_TOKENS=%q\n' "${closeout_max_new_tokens}" >> "${launch_env_path}"
  printf 'CLOSEOUT_POLL_SECONDS=%q\n' "${closeout_poll_seconds}" >> "${launch_env_path}"
  printf 'CLOSEOUT_TIMEOUT_SECONDS=%q\n' "${closeout_timeout_seconds}" >> "${launch_env_path}"
  if [[ -n "${closeout_binary_path}" ]]; then
    printf 'CLOSEOUT_BINARY_PATH=%q\n' "${closeout_binary_path}" >> "${launch_env_path}"
  fi
fi
if [[ "${attach_detached_score_closeout}" == "1" ]]; then
  printf 'SCORE_REPORT_PATH=%q\n' "${score_report_path}" >> "${launch_env_path}"
  printf 'SCORE_SUMMARY_PATH=%q\n' "${score_summary_path}" >> "${launch_env_path}"
  printf 'SCORE_LOG_PATH=%q\n' "${score_log_path}" >> "${launch_env_path}"
  printf 'SCORE_POLL_SECONDS=%q\n' "${score_poll_seconds}" >> "${launch_env_path}"
  printf 'SCORE_TIMEOUT_SECONDS=%q\n' "${score_timeout_seconds}" >> "${launch_env_path}"
  if [[ -n "${score_binary_path}" ]]; then
    printf 'SCORE_BINARY_PATH=%q\n' "${score_binary_path}" >> "${launch_env_path}"
  fi
fi

run_trainer() {
  cd "${repo_root}"
  run_env=(
    "TMPDIR=${tmpdir}"
    "CARGO_TARGET_DIR=${cargo_target_dir}"
  )
  if [[ -n "${challenge_max_steps}" ]]; then
    run_env+=("PSIONIC_PARAMETER_GOLF_HOMEGOLF_MAX_CHALLENGE_STEPS=${challenge_max_steps}")
  fi
  if [[ -n "${grad_clip_norm}" ]]; then
    run_env+=("PSIONIC_PARAMETER_GOLF_HOMEGOLF_GRAD_CLIP_NORM=${grad_clip_norm}")
  fi
  if [[ -n "${learning_rate_scale}" ]]; then
    run_env+=("PSIONIC_PARAMETER_GOLF_HOMEGOLF_LR_SCALE=${learning_rate_scale}")
  fi
  if [[ -n "${ema_decay}" ]]; then
    run_env+=("PSIONIC_PARAMETER_GOLF_EMA_DECAY=${ema_decay}")
  fi
  if [[ -n "${final_model_surface}" ]]; then
    run_env+=("PSIONIC_PARAMETER_GOLF_FINAL_MODEL_SURFACE=${final_model_surface}")
  fi
  env "${run_env[@]}" \
    "${trainer_command[@]}"
}

if [[ "${background}" == "1" ]]; then
  run_trainer >"${log_path}" 2>&1 &
  trainer_pid=$!
  printf '%s\n' "${trainer_pid}" > "${pid_path}"
  if [[ "${attach_prompt_closeout}" == "1" ]]; then
    closeout_command=(
      "${repo_root}/scripts/wait-parameter-golf-homegolf-prompt-closeout.sh"
      --report
      "${report_path}"
      --prompt
      "${closeout_prompt}"
      --max-new-tokens
      "${closeout_max_new_tokens}"
      --poll-seconds
      "${closeout_poll_seconds}"
      --timeout-seconds
      "${closeout_timeout_seconds}"
      --output
      "${prompt_report_path}"
      --summary
      "${closeout_summary_path}"
      --tmpdir
      "${tmpdir}"
      --cargo-target-dir
      "${cargo_target_dir}"
    )
    if [[ -n "${closeout_binary_path}" ]]; then
      closeout_command+=(
        --binary-path
        "${closeout_binary_path}"
      )
    fi
    (
      cd "${repo_root}"
      "${closeout_command[@]}"
    ) >"${prompt_log_path}" 2>&1 &
    prompt_pid=$!
    printf '%s\n' "${prompt_pid}" > "${prompt_pid_path}"
    printf 'PROMPT_WATCHER_PID=%q\n' "${prompt_pid}" >> "${launch_env_path}"
    printf 'PROMPT_PID_PATH=%q\n' "${prompt_pid_path}" >> "${launch_env_path}"
  fi
  if [[ "${attach_detached_score_closeout}" == "1" ]]; then
    score_closeout_command=(
      "${repo_root}/scripts/wait-parameter-golf-homegolf-artifact-score.sh"
      --report
      "${report_path}"
      --poll-seconds
      "${score_poll_seconds}"
      --timeout-seconds
      "${score_timeout_seconds}"
      --output
      "${score_report_path}"
      --summary
      "${score_summary_path}"
      --tmpdir
      "${tmpdir}"
      --cargo-target-dir
      "${cargo_target_dir}"
    )
    if [[ -n "${score_binary_path}" ]]; then
      score_closeout_command+=(
        --binary-path
        "${score_binary_path}"
      )
    fi
    (
      cd "${repo_root}"
      "${score_closeout_command[@]}"
    ) >"${score_log_path}" 2>&1 &
    score_pid=$!
    printf '%s\n' "${score_pid}" > "${score_pid_path}"
    printf 'SCORE_WATCHER_PID=%q\n' "${score_pid}" >> "${launch_env_path}"
    printf 'SCORE_PID_PATH=%q\n' "${score_pid_path}" >> "${launch_env_path}"
  fi
  echo "run_id=${run_id}"
  echo "executor=${executor_label}"
  echo "status=launched_background"
  echo "repo_root=${repo_root}"
  echo "output_root=${output_root}"
  echo "report_path=${report_path}"
  echo "log_path=${log_path}"
  echo "pid_path=${pid_path}"
  echo "tmpdir=${tmpdir}"
  echo "cargo_target_dir=${cargo_target_dir}"
  echo "pid=${trainer_pid}"
  if [[ "${attach_prompt_closeout}" == "1" ]]; then
    echo "prompt_report_path=${prompt_report_path}"
    echo "closeout_summary_path=${closeout_summary_path}"
    echo "prompt_log_path=${prompt_log_path}"
    echo "prompt_pid_path=${prompt_pid_path}"
    echo "prompt_pid=${prompt_pid}"
  fi
  if [[ "${attach_detached_score_closeout}" == "1" ]]; then
    echo "score_report_path=${score_report_path}"
    echo "score_summary_path=${score_summary_path}"
    echo "score_log_path=${score_log_path}"
    echo "score_pid_path=${score_pid_path}"
    echo "score_pid=${score_pid}"
  fi
  exit 0
fi

run_trainer 2>&1 | tee "${log_path}"

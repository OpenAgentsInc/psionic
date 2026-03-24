#!/usr/bin/env bash

set -euo pipefail

repo_root=""
log_path=""
python_bin="python3"
nproc_per_node="1"
run_id="baseline_sp1024"
data_path=""
tokenizer_path=""
max_wallclock_seconds="600"
val_loss_every="0"
extra_env=()

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-run-train-gpt-reference.sh --repo-root <path> --log <path> [options]

Options:
  --python <path>                   Python interpreter to use. Default: python3
  --run-id <id>                     RUN_ID for train_gpt.py. Default: baseline_sp1024
  --data-path <path>                DATA_PATH. Default: <repo-root>/data/datasets/fineweb10B_sp1024
  --tokenizer-path <path>           TOKENIZER_PATH. Default: <repo-root>/data/tokenizers/fineweb_1024_bpe.model
  --nproc-per-node <n>              torchrun local process count. Default: 1
  --max-wallclock-seconds <n>       MAX_WALLCLOCK_SECONDS. Default: 600
  --val-loss-every <n>              VAL_LOSS_EVERY. Default: 0
  --env <KEY=VALUE>                 Additional environment entry to pass through. Repeatable.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      repo_root="$2"
      shift 2
      ;;
    --log)
      log_path="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --data-path)
      data_path="$2"
      shift 2
      ;;
    --tokenizer-path)
      tokenizer_path="$2"
      shift 2
      ;;
    --nproc-per-node)
      nproc_per_node="$2"
      shift 2
      ;;
    --max-wallclock-seconds)
      max_wallclock_seconds="$2"
      shift 2
      ;;
    --val-loss-every)
      val_loss_every="$2"
      shift 2
      ;;
    --env)
      extra_env+=("$2")
      shift 2
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

if [[ -z "${repo_root}" || -z "${log_path}" ]]; then
  echo "error: --repo-root and --log are required" >&2
  usage
  exit 1
fi

repo_root="$(cd -- "${repo_root}" && pwd)"
if [[ -z "${data_path}" ]]; then
  data_path="${repo_root}/data/datasets/fineweb10B_sp1024"
fi
if [[ -z "${tokenizer_path}" ]]; then
  tokenizer_path="${repo_root}/data/tokenizers/fineweb_1024_bpe.model"
fi

if [[ ! -d "${repo_root}" ]]; then
  echo "error: repo root does not exist: ${repo_root}" >&2
  exit 1
fi
if [[ ! -f "${repo_root}/train_gpt.py" ]]; then
  echo "error: train_gpt.py is missing under ${repo_root}" >&2
  exit 1
fi
if [[ ! -d "${data_path}" ]]; then
  echo "error: DATA_PATH does not exist: ${data_path}" >&2
  exit 1
fi
if [[ ! -f "${tokenizer_path}" ]]; then
  echo "error: TOKENIZER_PATH does not exist: ${tokenizer_path}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${log_path}")"

cmd=(
  env
  "RUN_ID=${run_id}"
  "DATA_PATH=${data_path}"
  "TOKENIZER_PATH=${tokenizer_path}"
  "VOCAB_SIZE=1024"
  "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}"
  "VAL_LOSS_EVERY=${val_loss_every}"
)
for entry in "${extra_env[@]}"; do
  cmd+=("${entry}")
done
cmd+=(
  torchrun
  --standalone
  "--nproc_per_node=${nproc_per_node}"
  train_gpt.py
)

(
  cd "${repo_root}"
  "${python_bin}" -c "import sentencepiece, datasets, huggingface_hub, tqdm, torch" >/dev/null
  printf 'reference_run_start repo_root=%s run_id=%s data_path=%s tokenizer_path=%s nproc_per_node=%s max_wallclock_seconds=%s\n' \
    "${repo_root}" "${run_id}" "${data_path}" "${tokenizer_path}" "${nproc_per_node}" "${max_wallclock_seconds}"
  "${cmd[@]}"
) 2>&1 | tee "${log_path}"

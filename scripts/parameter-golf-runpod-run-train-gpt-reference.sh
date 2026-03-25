#!/usr/bin/env bash

set -euo pipefail

repo_root=""
log_path=""
python_bin="python3"
venv_dir=""
nproc_per_node="1"
run_id="baseline_sp1024"
data_path=""
tokenizer_path=""
max_wallclock_seconds="600"
val_loss_every="0"
bootstrap_venv="true"
extra_env=()

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-run-train-gpt-reference.sh --repo-root <path> --log <path> [options]

Options:
  --python <path>                   Python interpreter to use. Default: python3
  --venv <path>                     Virtualenv directory. Default: <repo-root>/.venv
  --run-id <id>                     RUN_ID for train_gpt.py. Default: baseline_sp1024
  --data-path <path>                DATA_PATH. Default: <repo-root>/data/datasets/fineweb10B_sp1024
  --tokenizer-path <path>           TOKENIZER_PATH. Default: <repo-root>/data/tokenizers/fineweb_1024_bpe.model
  --nproc-per-node <n>              torchrun local process count. Default: 1
  --max-wallclock-seconds <n>       MAX_WALLCLOCK_SECONDS. Default: 600
  --val-loss-every <n>              VAL_LOSS_EVERY. Default: 0
  --no-bootstrap-venv               Refuse instead of creating/upgrading the venv when the current Python lacks train_gpt.py support.
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
    --venv)
      venv_dir="$2"
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
    --no-bootstrap-venv)
      bootstrap_venv="false"
      shift
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
if [[ -z "${venv_dir}" ]]; then
  venv_dir="${repo_root}/.venv"
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

python_supports_train_gpt() {
  local candidate="$1"
  "${candidate}" - <<'PY' >/dev/null 2>&1
import importlib

for module_name in (
    "sentencepiece",
    "datasets",
    "huggingface_hub",
    "kernels",
    "tqdm",
    "torch",
    "tiktoken",
):
    importlib.import_module(module_name)

import torch
import torch.nn.functional as F

q = torch.randn(1, 1, 1, 8)
k = torch.randn(1, 1, 1, 8)
v = torch.randn(1, 1, 1, 8)
F.scaled_dot_product_attention(q, k, v, enable_gqa=False)
PY
}

python_torch_version() {
  local candidate="$1"
  "${candidate}" - <<'PY'
import torch
print(torch.__version__)
PY
}

resolved_python=""

ensure_train_gpt_python() {
  local candidate="${python_bin}"
  if [[ -x "${venv_dir}/bin/python" ]]; then
    candidate="${venv_dir}/bin/python"
  fi
  if python_supports_train_gpt "${candidate}"; then
    printf 'reference_python_ready executable=%s torch_version=%s bootstrap=not_needed\n' \
      "${candidate}" "$(python_torch_version "${candidate}")"
    resolved_python="${candidate}"
    return 0
  fi
  if [[ "${bootstrap_venv}" != "true" ]]; then
    echo "error: python ${candidate} does not satisfy the current train_gpt.py dependency contract and --no-bootstrap-venv was supplied" >&2
    exit 1
  fi
  if [[ ! -d "${venv_dir}" ]]; then
    "${python_bin}" -m venv "${venv_dir}"
  fi
  "${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
  "${venv_dir}/bin/pip" install --upgrade -r "${repo_root}/requirements.txt"
  candidate="${venv_dir}/bin/python"
  if ! python_supports_train_gpt "${candidate}"; then
    echo "error: bootstrapped virtualenv still does not satisfy the current train_gpt.py dependency contract" >&2
    exit 1
  fi
  printf 'reference_python_ready executable=%s torch_version=%s bootstrap=performed\n' \
    "${candidate}" "$(python_torch_version "${candidate}")"
  resolved_python="${candidate}"
}

ensure_train_gpt_python

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
  "${resolved_python}"
  -m
  torch.distributed.run
  --standalone
  "--nproc_per_node=${nproc_per_node}"
  train_gpt.py
)

(
  cd "${repo_root}"
  printf 'reference_run_start repo_root=%s run_id=%s data_path=%s tokenizer_path=%s nproc_per_node=%s max_wallclock_seconds=%s\n' \
    "${repo_root}" "${run_id}" "${data_path}" "${tokenizer_path}" "${nproc_per_node}" "${max_wallclock_seconds}"
  "${cmd[@]}"
) 2>&1 | tee "${log_path}"

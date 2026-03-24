#!/usr/bin/env bash

set -euo pipefail

contract_path=""
target_root=""

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-materialize-public-cache.sh --contract <path> --target-root <path>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --contract)
      contract_path="$2"
      shift 2
      ;;
    --target-root)
      target_root="$2"
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

if [[ -z "${contract_path}" || -z "${target_root}" ]]; then
  usage
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

compute_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

git_clone_url="$(jq -r '.parameter_golf_repo.clone_url' "${contract_path}")"
git_revision="$(jq -r '.parameter_golf_repo.git_revision' "${contract_path}")"
variant="$(jq -r '.dataset.variant' "${contract_path}")"
train_shards="$(jq -r '.dataset.train_shards' "${contract_path}")"
dataset_relative_root="$(jq -r '.dataset.local_relative_root' "${contract_path}")"
tokenizer_relative_path="$(jq -r '.tokenizer.local_relative_path' "${contract_path}")"
expected_tokenizer_sha256="$(jq -r '.tokenizer.sha256' "${contract_path}")"
matched_fineweb_repo_id="$(jq -r '.matched_fineweb.repo_id // empty' "${contract_path}")"
matched_fineweb_remote_root_prefix="$(
  jq -r '.matched_fineweb.remote_root_prefix // empty' "${contract_path}"
)"

mkdir -p "$(dirname "${target_root}")"
if [[ ! -d "${target_root}/.git" ]]; then
  git clone "${git_clone_url}" "${target_root}"
fi
git -C "${target_root}" fetch --depth=1 origin "${git_revision}"
git -C "${target_root}" checkout --detach "${git_revision}"

if [[ ! -d "${target_root}/.venv" ]]; then
  python3 -m venv "${target_root}/.venv"
fi

# shellcheck disable=SC1091
source "${target_root}/.venv/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install huggingface-hub >/dev/null

if [[ -n "${matched_fineweb_repo_id}" ]]; then
  export MATCHED_FINEWEB_REPO_ID="${matched_fineweb_repo_id}"
fi
if [[ -n "${matched_fineweb_remote_root_prefix}" ]]; then
  export MATCHED_FINEWEB_REMOTE_ROOT_PREFIX="${matched_fineweb_remote_root_prefix}"
fi

(
  cd "${target_root}"
  python data/cached_challenge_fineweb.py --variant "${variant}" --train-shards "${train_shards}"
)

dataset_root="${target_root}/${dataset_relative_root}"
tokenizer_path="${target_root}/${tokenizer_relative_path}"

if [[ ! -d "${dataset_root}" ]]; then
  echo "error: expected dataset root ${dataset_root} after cache materialization" >&2
  exit 1
fi
if [[ ! -f "${tokenizer_path}" ]]; then
  echo "error: expected tokenizer path ${tokenizer_path} after cache materialization" >&2
  exit 1
fi

actual_tokenizer_sha256="$(compute_sha256 "${tokenizer_path}")"
if [[ "${actual_tokenizer_sha256}" != "${expected_tokenizer_sha256}" ]]; then
  echo "error: tokenizer sha256 mismatch; expected ${expected_tokenizer_sha256}, found ${actual_tokenizer_sha256}" >&2
  exit 1
fi

jq -n \
  --arg target_root "${target_root}" \
  --arg git_revision "${git_revision}" \
  --arg dataset_root "${dataset_root}" \
  --arg tokenizer_path "${tokenizer_path}" \
  --arg tokenizer_sha256 "${actual_tokenizer_sha256}" \
  '{
    materialized: true,
    target_root: $target_root,
    parameter_golf_git_revision: $git_revision,
    dataset_root: $dataset_root,
    tokenizer_path: $tokenizer_path,
    tokenizer_sha256: $tokenizer_sha256
  }'

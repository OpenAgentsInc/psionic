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

compute_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

json_field() {
  local query="$1"
  python3 - "${contract_path}" "${query}" <<'PY'
import json
import sys
from pathlib import Path

contract_path = Path(sys.argv[1])
query = sys.argv[2].strip()
empty_ok = False
if query.endswith("// empty"):
    query = query[:-9].strip()
    empty_ok = True
if not query.startswith("."):
    raise SystemExit(f"unsupported json query: {query}")
parts = [part for part in query.split(".") if part]
with contract_path.open("r", encoding="utf-8") as handle:
    current = json.load(handle)
for part in parts:
    if isinstance(current, dict) and part in current:
        current = current[part]
    else:
        if empty_ok:
            print("")
            raise SystemExit(0)
        raise SystemExit(f"missing contract path: {query}")
if current is None and empty_ok:
    print("")
elif isinstance(current, bool):
    print("true" if current else "false")
else:
    print(current)
PY
}

git_clone_url="$(json_field '.parameter_golf_repo.clone_url')"
git_revision="$(json_field '.parameter_golf_repo.git_revision')"
variant="$(json_field '.dataset.variant')"
train_shards="$(json_field '.dataset.train_shards')"
dataset_relative_root="$(json_field '.dataset.local_relative_root')"
tokenizer_relative_path="$(json_field '.tokenizer.local_relative_path')"
expected_tokenizer_sha256="$(json_field '.tokenizer.sha256')"
matched_fineweb_repo_id="$(json_field '.matched_fineweb.repo_id // empty')"
matched_fineweb_remote_root_prefix="$(json_field '.matched_fineweb.remote_root_prefix // empty')"
dataset_root="${target_root}/${dataset_relative_root}"
tokenizer_path="${target_root}/${tokenizer_relative_path}"
workspace_checkout_posture="cloned_git_checkout"

mkdir -p "$(dirname "${target_root}")"
if [[ -d "${target_root}/.git" ]]; then
  workspace_checkout_posture="existing_git_checkout"
elif [[ -d "${target_root}" ]] && find "${target_root}" -mindepth 1 -print -quit >/dev/null 2>&1; then
  workspace_checkout_posture="existing_non_git_target"
else
  git clone "${git_clone_url}" "${target_root}"
fi
if [[ -d "${target_root}/.git" ]]; then
  git -C "${target_root}" fetch --depth=1 origin "${git_revision}"
  git -C "${target_root}" checkout --detach "${git_revision}"
fi

if [[ ! -d "${target_root}/.venv" ]]; then
  python3 -m venv "${target_root}/.venv"
fi

# shellcheck disable=SC1091
source "${target_root}/.venv/bin/activate"
if ! python -c 'import huggingface_hub' >/dev/null 2>&1; then
  python -m pip install --upgrade pip >/dev/null
  python -m pip install huggingface-hub >/dev/null
fi

if [[ -n "${matched_fineweb_repo_id}" ]]; then
  export MATCHED_FINEWEB_REPO_ID="${matched_fineweb_repo_id}"
fi
if [[ -n "${matched_fineweb_remote_root_prefix}" ]]; then
  export MATCHED_FINEWEB_REMOTE_ROOT_PREFIX="${matched_fineweb_remote_root_prefix}"
fi

bootstrap_entrypoint="${target_root}/data/cached_challenge_fineweb.py"
if [[ -f "${bootstrap_entrypoint}" ]]; then
  (
    cd "${target_root}"
    python data/cached_challenge_fineweb.py --variant "${variant}" --train-shards "${train_shards}"
  )
elif [[ "${workspace_checkout_posture}" == "existing_non_git_target" ]] \
  && [[ -d "${dataset_root}" ]] \
  && [[ -f "${tokenizer_path}" ]]; then
  workspace_checkout_posture="existing_non_git_cache_only_target"
else
  echo "error: expected bootstrap entrypoint ${bootstrap_entrypoint} or an already materialized cache target" >&2
  exit 1
fi

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

python3 - "${target_root}" "${git_revision}" "${dataset_root}" "${tokenizer_path}" "${actual_tokenizer_sha256}" "${workspace_checkout_posture}" <<'PY'
import json
import sys

target_root, git_revision, dataset_root, tokenizer_path, tokenizer_sha256, workspace_checkout_posture = sys.argv[1:]
print(
    json.dumps(
        {
            "materialized": True,
            "target_root": target_root,
            "parameter_golf_git_revision": git_revision,
            "workspace_checkout_posture": workspace_checkout_posture,
            "dataset_root": dataset_root,
            "tokenizer_path": tokenizer_path,
            "tokenizer_sha256": tokenizer_sha256,
        },
        indent=2,
    )
)
PY

#!/usr/bin/env bash

set -euo pipefail

LOG_FILE="/var/log/psion-google-startup.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

metadata_value() {
  local key="$1"
  curl -fsSL -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}"
}

ensure_rust_toolchain() {
  local toolchain="$1"
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain "${toolchain}"
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"
}

check_free_disk_gb() {
  local path="$1"
  local low_watermark_gb="$2"
  local available_gb
  available_gb="$(df --output=avail -BG "${path}" | tail -1 | tr -dc '0-9')"
  if [[ -z "${available_gb}" ]]; then
    echo "error: failed to determine free disk space for ${path}" >&2
    exit 1
  fi
  if (( available_gb < low_watermark_gb )); then
    echo "error: free disk ${available_gb}GB is below low watermark ${low_watermark_gb}GB" >&2
    exit 1
  fi
}

main() {
  local run_id bucket_url repo_clone_url git_revision training_command workspace_root
  local output_subdir log_subdir scratch_subdir repo_subdir low_disk_watermark_gb toolchain
  local apt_packages
  local run_root output_dir log_dir scratch_dir repo_dir entrypoint_file

  run_id="$(metadata_value psion-run-id)"
  bucket_url="$(metadata_value psion-bucket-url)"
  repo_clone_url="$(metadata_value psion-repo-clone-url)"
  git_revision="$(metadata_value psion-git-revision)"
  training_command="$(metadata_value psion-training-command)"
  workspace_root="$(metadata_value psion-workspace-root)"
  output_subdir="$(metadata_value psion-output-subdir)"
  log_subdir="$(metadata_value psion-log-subdir)"
  scratch_subdir="$(metadata_value psion-scratch-subdir)"
  repo_subdir="$(metadata_value psion-repo-subdir)"
  low_disk_watermark_gb="$(metadata_value psion-low-disk-watermark-gb)"
  toolchain="$(metadata_value psion-rust-toolchain)"
  apt_packages="$(metadata_value psion-apt-packages)"

  echo "psion google startup: run=${run_id} repo=${repo_clone_url} revision=${git_revision}"
  echo "psion google startup: bucket=${bucket_url}"

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  # shellcheck disable=SC2086
  apt-get install -y ${apt_packages}

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not found on the selected GPU image" >&2
    exit 1
  fi
  nvidia-smi

  ensure_rust_toolchain "${toolchain}"
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"

  run_root="${workspace_root}/runs/${run_id}"
  output_dir="${run_root}/${output_subdir}"
  log_dir="${run_root}/${log_subdir}"
  scratch_dir="${run_root}/${scratch_subdir}"
  repo_dir="${run_root}/${repo_subdir}"
  mkdir -p "${output_dir}" "${log_dir}" "${scratch_dir}"

  check_free_disk_gb "${run_root}" "${low_disk_watermark_gb}"

  git clone "${repo_clone_url}" "${repo_dir}"
  git -C "${repo_dir}" fetch --depth=1 origin "${git_revision}"
  git -C "${repo_dir}" checkout --detach "${git_revision}"

  check_free_disk_gb "${run_root}" "${low_disk_watermark_gb}"

  export PSION_RUN_ID="${run_id}"
  export PSION_BUCKET_URL="${bucket_url}"
  export PSION_WORKSPACE_ROOT="${run_root}"
  export PSION_OUTPUT_DIR="${output_dir}"
  export PSION_LOG_DIR="${log_dir}"
  export PSION_SCRATCH_DIR="${scratch_dir}"

  entrypoint_file="${run_root}/psion_google_training_entrypoint.sh"
  cat > "${entrypoint_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.cargo/env"
export PSION_RUN_ID="${PSION_RUN_ID}"
export PSION_BUCKET_URL="${PSION_BUCKET_URL}"
export PSION_WORKSPACE_ROOT="${PSION_WORKSPACE_ROOT}"
export PSION_OUTPUT_DIR="${PSION_OUTPUT_DIR}"
export PSION_LOG_DIR="${PSION_LOG_DIR}"
export PSION_SCRATCH_DIR="${PSION_SCRATCH_DIR}"
cd "${repo_dir}"
${training_command}
EOF
  chmod +x "${entrypoint_file}"

  echo "psion google startup: launching training command"
  nohup "${entrypoint_file}" \
    > "${log_dir}/training.stdout.log" \
    2> "${log_dir}/training.stderr.log" &
  echo $! > "${run_root}/training.pid"
  echo "psion google startup: pid=$(cat "${run_root}/training.pid")"
}

main "$@"

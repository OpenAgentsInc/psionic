#!/usr/bin/env bash

set -Eeuo pipefail

LOG_FILE="/var/log/psion-google-two-node-swarm-startup.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
export HOME="${HOME:-/root}"

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

metadata_value() {
  local key="$1"
  curl -fsSL -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}"
}

metadata_value_optional() {
  local key="$1"
  curl -fsSL -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}" 2>/dev/null || true
}

compute_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

wait_for_object() {
  local object_path="$1"
  local timeout_seconds="$2"
  local poll_seconds="$3"
  local started_at
  started_at="$(date +%s)"
  while true; do
    if gcloud storage ls "${object_path}" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - started_at >= timeout_seconds )); then
      echo "error: object ${object_path} did not become visible within ${timeout_seconds}s" >&2
      exit 1
    fi
    sleep "${poll_seconds}"
  done
}

ensure_rust_toolchain() {
  local toolchain="$1"
  if command -v cargo >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "${HOME}/.cargo/env" 2>/dev/null || true
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

clone_or_refresh_repo() {
  local repo_clone_url="$1"
  local git_revision="$2"
  local repo_dir="$3"
  if [[ -d "${repo_dir}/.git" ]]; then
    git -C "${repo_dir}" fetch --depth=1 origin "${git_revision}"
  else
    rm -rf "${repo_dir}"
    git clone --filter=blob:none "${repo_clone_url}" "${repo_dir}"
  fi
  git -C "${repo_dir}" checkout --force "${git_revision}"
}

RUN_ID="$(metadata_value "psion-run-id")"
PSION_SWARM_ROLE="$(metadata_value "psion-swarm-role")"
PSION_ROLE_ID="$(metadata_value "psion-role-id")"
PSION_NODE_ID="$(metadata_value "psion-node-id")"
PSION_BUCKET_URL="$(metadata_value "psion-bucket-url")"
PSION_REPO_CLONE_URL="$(metadata_value "psion-repo-clone-url")"
PSION_GIT_REVISION="$(metadata_value "psion-git-revision")"
PSION_WORKSPACE_ROOT="$(metadata_value "psion-workspace-root")"
PSION_RUST_TOOLCHAIN="$(metadata_value "psion-rust-toolchain")"
PSION_CLUSTER_MANIFEST_URI="$(metadata_value "psion-cluster-manifest-uri")"
PSION_LOCAL_ENDPOINT_URI="$(metadata_value "psion-local-endpoint-uri")"
PSION_PEER_ENDPOINT_URI="$(metadata_value "psion-peer-endpoint-uri")"
PSION_RUNTIME_REPORT_URI="$(metadata_value "psion-runtime-report-uri")"
PSION_SELECTED_IMPAIRMENT_PROFILE_ID="$(metadata_value "psion-selected-impairment-profile-id")"
PSION_ENDPOINT_MANIFEST_TIMEOUT_SECONDS="${PSION_ENDPOINT_MANIFEST_TIMEOUT_SECONDS:-$(metadata_value "psion-endpoint-manifest-timeout-seconds")}"
PSION_ENDPOINT_MANIFEST_POLL_INTERVAL_SECONDS="${PSION_ENDPOINT_MANIFEST_POLL_INTERVAL_SECONDS:-$(metadata_value "psion-endpoint-manifest-poll-interval-seconds")}"
PSION_LOW_DISK_WATERMARK_GB="${PSION_LOW_DISK_WATERMARK_GB:-$(metadata_value "psion-low-disk-watermark-gb")}"
PSION_APT_PACKAGES="$(metadata_value_optional "psion-apt-packages" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
PSION_PRE_TRAINING_COMMAND="$(metadata_value_optional "psion-pre-training-command")"
PSION_TRAINING_COMMAND="$(metadata_value "psion-training-command")"

RUN_ROOT="${PSION_WORKSPACE_ROOT}/runs/${RUN_ID}/${PSION_SWARM_ROLE}"
SCRATCH_DIR="${RUN_ROOT}/scratch"
REPO_DIR="${RUN_ROOT}/repo"
export PSION_RUN_ROOT="${RUN_ROOT}"
export PSION_SCRATCH_DIR="${SCRATCH_DIR}"
export PSION_REPO_DIR="${REPO_DIR}"
export PSION_RUNTIME_REPORT_FILE="${SCRATCH_DIR}/psion_google_two_node_swarm_runtime_report.json"
export PSION_CLUSTER_MANIFEST_FILE="${SCRATCH_DIR}/psion_google_two_node_swarm_cluster_manifest.json"
export PSION_LOCAL_ENDPOINT_FILE="${SCRATCH_DIR}/psion_google_two_node_swarm_local_endpoint_manifest.json"
export PSION_PEER_ENDPOINT_FILE="${SCRATCH_DIR}/psion_google_two_node_swarm_peer_endpoint_manifest.json"

mkdir -p "${SCRATCH_DIR}"
check_free_disk_gb "/" "${PSION_LOW_DISK_WATERMARK_GB}"

if [[ -n "${PSION_APT_PACKAGES}" ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  # shellcheck disable=SC2086
  apt-get install -y ${PSION_APT_PACKAGES}
fi

ensure_rust_toolchain "${PSION_RUST_TOOLCHAIN}"
clone_or_refresh_repo "${PSION_REPO_CLONE_URL}" "${PSION_GIT_REVISION}" "${REPO_DIR}"

wait_for_object "${PSION_CLUSTER_MANIFEST_URI}" "${PSION_ENDPOINT_MANIFEST_TIMEOUT_SECONDS}" "${PSION_ENDPOINT_MANIFEST_POLL_INTERVAL_SECONDS}"
wait_for_object "${PSION_LOCAL_ENDPOINT_URI}" "${PSION_ENDPOINT_MANIFEST_TIMEOUT_SECONDS}" "${PSION_ENDPOINT_MANIFEST_POLL_INTERVAL_SECONDS}"
wait_for_object "${PSION_PEER_ENDPOINT_URI}" "${PSION_ENDPOINT_MANIFEST_TIMEOUT_SECONDS}" "${PSION_ENDPOINT_MANIFEST_POLL_INTERVAL_SECONDS}"

gcloud storage cp --quiet "${PSION_CLUSTER_MANIFEST_URI}" "${PSION_CLUSTER_MANIFEST_FILE}" >/dev/null
gcloud storage cp --quiet "${PSION_LOCAL_ENDPOINT_URI}" "${PSION_LOCAL_ENDPOINT_FILE}" >/dev/null
gcloud storage cp --quiet "${PSION_PEER_ENDPOINT_URI}" "${PSION_PEER_ENDPOINT_FILE}" >/dev/null

echo "psion google two-node swarm startup: run=${RUN_ID} role=${PSION_SWARM_ROLE} node=${PSION_NODE_ID}"
echo "psion google two-node swarm startup: impairment_profile=${PSION_SELECTED_IMPAIRMENT_PROFILE_ID}"
echo "psion google two-node swarm startup: manifest_sha256=$(compute_sha256 "${PSION_CLUSTER_MANIFEST_FILE}")"

cd "${REPO_DIR}"

if [[ -n "${PSION_PRE_TRAINING_COMMAND}" ]]; then
  echo "psion google two-node swarm startup: pre-training command begins at $(timestamp_utc)"
  eval "${PSION_PRE_TRAINING_COMMAND}"
fi

echo "psion google two-node swarm startup: training command begins at $(timestamp_utc)"
eval "${PSION_TRAINING_COMMAND}"

if [[ ! -f "${PSION_RUNTIME_REPORT_FILE}" ]]; then
  echo "error: runtime report ${PSION_RUNTIME_REPORT_FILE} was not written" >&2
  exit 1
fi

gcloud storage cp --quiet "${PSION_RUNTIME_REPORT_FILE}" "${PSION_RUNTIME_REPORT_URI}" >/dev/null
wait_for_object "${PSION_RUNTIME_REPORT_URI}" 120 2

echo "psion google two-node swarm startup: uploaded runtime report ${PSION_RUNTIME_REPORT_URI}"

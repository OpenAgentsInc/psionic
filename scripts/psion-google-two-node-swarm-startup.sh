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

free_disk_gb() {
  local path="$1"
  df --output=avail -BG "${path}" | tail -1 | tr -dc '0-9'
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
export PSION_BRINGUP_REPORT_FILE="${SCRATCH_DIR}/psion_google_two_node_swarm_bringup_report.json"

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

local_node_json="$(jq -c --arg role_id "${PSION_ROLE_ID}" '.nodes[] | select(.role_id == $role_id)' "${PSION_CLUSTER_MANIFEST_FILE}")"
if [[ -z "${local_node_json}" ]]; then
  echo "error: cluster manifest did not contain node role ${PSION_ROLE_ID}" >&2
  exit 1
fi
local_endpoint_value="$(jq -r '.endpoint' "${PSION_LOCAL_ENDPOINT_FILE}")"
peer_endpoint_value="$(jq -r '.endpoint' "${PSION_PEER_ENDPOINT_FILE}")"
bringup_report_uri="$(jq -r '.bringup_report_uri' <<<"${local_node_json}")"
available_gb="$(free_disk_gb "/")"
gpu_inventory_json='[]'
gpu_count=0
cuda_driver_ready=false
cuda_version=""
driver_version=""
if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_inventory_json="$(
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | \
      python3 -c 'import csv, json, sys
rows = []
for row in csv.reader(sys.stdin):
    if not row:
        continue
    rows.append({
        "name": row[0].strip(),
        "memory_total_mib": int(float(row[1].strip())),
        "driver_version": row[2].strip(),
    })
print(json.dumps(rows))'
  )"; then
    gpu_count="$(jq 'length' <<<"${gpu_inventory_json}")"
    if (( gpu_count > 0 )); then
      cuda_driver_ready=true
      driver_version="$(jq -r '.[0].driver_version' <<<"${gpu_inventory_json}")"
      cuda_version="$(nvidia-smi | sed -n '1,3p' | sed -n 's/.*CUDA Version: \\([0-9.]*\\).*/\\1/p' | head -n 1)"
    fi
  fi
fi

bringup_status="ready"
bringup_detail="The node booted on the admitted Google swarm machine shape, downloaded the exact manifest set, exposed the expected private endpoint, and saw healthy CUDA inventory before the bounded runtime command began."
expected_accelerator_count="$(jq -r '.accelerator_count' <<<"${local_node_json}")"
if [[ "${cuda_driver_ready}" != "true" || "${gpu_count}" -lt "${expected_accelerator_count}" ]]; then
  bringup_status="refused_machine_contract"
  bringup_detail="The node did not expose the admitted CUDA inventory for the Google swarm lane, so startup refuses the bounded runtime command instead of fabricating a healthy bring-up."
fi

jq -n \
  --arg schema_version "psion.google_two_node_swarm_bringup_report.v1" \
  --arg created_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --arg node_id "${PSION_NODE_ID}" \
  --arg role_id "${PSION_ROLE_ID}" \
  --arg runtime_role "${PSION_SWARM_ROLE}" \
  --arg selected_impairment_profile_id "${PSION_SELECTED_IMPAIRMENT_PROFILE_ID}" \
  --arg git_revision "${PSION_GIT_REVISION}" \
  --arg cluster_manifest_uri "${PSION_CLUSTER_MANIFEST_URI}" \
  --arg local_endpoint_uri "${PSION_LOCAL_ENDPOINT_URI}" \
  --arg peer_endpoint_uri "${PSION_PEER_ENDPOINT_URI}" \
  --arg local_endpoint "${local_endpoint_value}" \
  --arg peer_endpoint "${peer_endpoint_value}" \
  --arg machine_type "$(jq -r '.machine_type' <<<"${local_node_json}")" \
  --arg accelerator_type "$(jq -r '.accelerator_type' <<<"${local_node_json}")" \
  --argjson accelerator_count "${expected_accelerator_count}" \
  --arg hostname_short "$(hostname -s)" \
  --arg hostname_fqdn "$(hostname -f 2>/dev/null || hostname)" \
  --arg kernel_release "$(uname -r)" \
  --arg os_summary "$(uname -sm)" \
  --arg workspace_root "${PSION_WORKSPACE_ROOT}" \
  --arg run_root "${RUN_ROOT}" \
  --arg scratch_dir "${SCRATCH_DIR}" \
  --arg bringup_status "${bringup_status}" \
  --arg bringup_detail "${bringup_detail}" \
  --arg cuda_version "${cuda_version}" \
  --arg driver_version "${driver_version}" \
  --argjson cuda_driver_ready "${cuda_driver_ready}" \
  --argjson gpu_count "${gpu_count}" \
  --argjson gpu_inventory "${gpu_inventory_json}" \
  --argjson available_gb "${available_gb:-0}" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    node_id: $node_id,
    role_id: $role_id,
    runtime_role: $runtime_role,
    selected_impairment_profile_id: $selected_impairment_profile_id,
    git_revision: $git_revision,
    manifest_inputs: {
      cluster_manifest_uri: $cluster_manifest_uri,
      local_endpoint_uri: $local_endpoint_uri,
      peer_endpoint_uri: $peer_endpoint_uri
    },
    cluster_endpoints: {
      local_endpoint: $local_endpoint,
      peer_endpoint: $peer_endpoint
    },
    machine: {
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: $accelerator_count,
      hostname_short: $hostname_short,
      hostname_fqdn: $hostname_fqdn,
      kernel_release: $kernel_release,
      os_summary: $os_summary
    },
    scratch_posture: {
      workspace_root: $workspace_root,
      run_root: $run_root,
      scratch_dir: $scratch_dir,
      available_gb: $available_gb
    },
    cuda_posture: {
      driver_ready: $cuda_driver_ready,
      driver_version: (if $driver_version == "" then null else $driver_version end),
      cuda_version: (if $cuda_version == "" then null else $cuda_version end),
      gpu_count: $gpu_count,
      gpu_inventory: $gpu_inventory
    },
    status: $bringup_status,
    detail: $bringup_detail,
    claim_boundary: "This bring-up report proves node-local machine facts, private endpoint visibility, scratch posture, and CUDA inventory for the bounded Google two-node swarm lane. It does not by itself prove successful cluster membership, contributor execution, validator acceptance, or aggregation."
  }' > "${PSION_BRINGUP_REPORT_FILE}"

gcloud storage cp --quiet "${PSION_BRINGUP_REPORT_FILE}" "${bringup_report_uri}" >/dev/null
wait_for_object "${bringup_report_uri}" 120 2

echo "psion google two-node swarm startup: run=${RUN_ID} role=${PSION_SWARM_ROLE} node=${PSION_NODE_ID}"
echo "psion google two-node swarm startup: impairment_profile=${PSION_SELECTED_IMPAIRMENT_PROFILE_ID}"
echo "psion google two-node swarm startup: manifest_sha256=$(compute_sha256 "${PSION_CLUSTER_MANIFEST_FILE}")"

if [[ "${bringup_status}" != "ready" ]]; then
  echo "error: bring-up refused machine contract for ${PSION_NODE_ID}" >&2
  exit 1
fi

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

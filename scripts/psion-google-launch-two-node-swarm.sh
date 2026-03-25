#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
PAIR_ID="${PAIR_ID:-}"
RUN_ID="${RUN_ID:-}"
COORDINATOR_INSTANCE_NAME="${COORDINATOR_INSTANCE_NAME:-}"
CONTRIBUTOR_INSTANCE_NAME="${CONTRIBUTOR_INSTANCE_NAME:-}"
IMPAIRMENT_PROFILE_ID="${IMPAIRMENT_PROFILE_ID:-clean_baseline}"
MANIFEST_ONLY=false

usage() {
  cat <<'EOF'
Usage: psion-google-launch-two-node-swarm.sh [options]

Options:
  --pair <pair_id>                         Force one admitted zone pair instead of auto-selecting the first ready pair.
  --run-id <run_id>                       Override the generated run id.
  --coordinator-instance-name <name>      Override the coordinator instance name.
  --contributor-instance-name <name>      Override the contributor instance name.
  --impairment-profile <profile_id>       Select one admitted impairment profile.
  --manifest-only                         Materialize and upload the launch artifacts without creating either VM.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pair)
      PAIR_ID="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --coordinator-instance-name)
      COORDINATOR_INSTANCE_NAME="$2"
      shift 2
      ;;
    --contributor-instance-name)
      CONTRIBUTOR_INSTANCE_NAME="$2"
      shift 2
      ;;
    --impairment-profile)
      IMPAIRMENT_PROFILE_ID="$2"
      shift 2
      ;;
    --manifest-only)
      MANIFEST_ONLY=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
LAUNCH_FILE="${LAUNCH_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_two_node_swarm_launch_profiles_v1.json}"
CONTRACT_FILE="${CONTRACT_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_two_node_swarm_contract_v1.json}"
NETWORK_POSTURE_FILE="${NETWORK_POSTURE_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_two_node_swarm_network_posture_v1.json}"
IDENTITY_PROFILE_FILE="${IDENTITY_PROFILE_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_two_node_swarm_identity_profile_v1.json}"
QUOTA_PREFLIGHT="${QUOTA_PREFLIGHT:-${REPO_ROOT}/scripts/psion-google-quota-preflight-two-node-swarm.sh}"
STARTUP_SCRIPT="${STARTUP_SCRIPT:-${REPO_ROOT}/scripts/psion-google-two-node-swarm-startup.sh}"

for required_command in gcloud jq git python3; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "error: ${required_command} is required" >&2
    exit 1
  fi
done

compute_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

sanitize_name() {
  tr '[:upper:]_' '[:lower:]-' <<<"$1" | tr -cd 'a-z0-9-' | sed 's/^-*//; s/-*$//'
}

make_instance_name() {
  local base="$1"
  local suffix="$2"
  local raw
  raw="$(sanitize_name "${base}-${suffix}")"
  raw="${raw:0:63}"
  raw="${raw%-}"
  if [[ -z "${raw}" ]]; then
    echo "error: failed to derive an instance name from ${base}-${suffix}" >&2
    exit 1
  fi
  printf '%s' "${raw}"
}

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

wait_for_object() {
  local object_path="$1"
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if gcloud storage ls "${object_path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: object ${object_path} did not become visible" >&2
  exit 1
}

compute_admission_token() {
  python3 - "$1" "$2" <<'PY'
import hashlib
import sys

run_id = sys.argv[1]
contract_digest = sys.argv[2]
print(hashlib.sha256(f"psion-google-two-node-swarm-admission|{run_id}|{contract_digest}".encode("utf-8")).hexdigest())
PY
}

compute_cluster_id() {
  python3 - "$1" "$2" "$3" <<'PY'
import hashlib
import sys

run_id = sys.argv[1]
contract_digest = sys.argv[2]
namespace = sys.argv[3]
admission = hashlib.sha256(f"psion-google-two-node-swarm-admission|{run_id}|{contract_digest}".encode("utf-8")).hexdigest()
payload = namespace.encode("utf-8") + b"\0" + admission.encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

profile_json() {
  local profile_id="$1"
  jq -c --arg profile_id "${profile_id}" '.profiles[] | select(.profile_id == $profile_id)' "${LAUNCH_FILE}"
}

identity_node_json() {
  local role_id="$1"
  jq -c --arg role_id "${role_id}" '.node_roles[] | select(.role_id == $role_id)' "${IDENTITY_PROFILE_FILE}"
}

contract_node_json() {
  local role_id="$1"
  jq -c --arg role_id "${role_id}" '.nodes[] | select(.role_id == $role_id)' "${CONTRACT_FILE}"
}

subnetwork_for_role() {
  local role_id="$1"
  jq -r --arg role_id "${role_id}" '.subnetworks[] | select(.role_id == $role_id) | .subnetwork_name' "${NETWORK_POSTURE_FILE}"
}

target_tags_for_role() {
  local role_id="$1"
  jq -r --arg role_id "${role_id}" '.node_roles[] | select(.role_id == $role_id) | .target_tags | join(",")' "${IDENTITY_PROFILE_FILE}"
}

labels_for_role() {
  local role_id="$1"
  jq -r --arg role_id "${role_id}" '
    (.recommended_instance_labels + (.node_roles[] | select(.role_id == $role_id) | .instance_labels))
    | to_entries
    | map("\(.key | gsub("_"; "-"))=\(.value | tostring | ascii_downcase | gsub("[^a-z0-9-]"; "-"))")
    | join(",")
  ' "${IDENTITY_PROFILE_FILE}"
}

wait_for_internal_ip() {
  local instance_name="$1"
  local zone="$2"
  local ip=""
  local attempt
  for attempt in $(seq 1 90); do
    ip="$(
      gcloud compute instances describe "${instance_name}" \
        --project="${PROJECT_ID}" \
        --zone="${zone}" \
        --format='get(networkInterfaces[0].networkIP)' 2>/dev/null || true
    )"
    if [[ -n "${ip}" ]]; then
      printf '%s' "${ip}"
      return 0
    fi
    sleep 2
  done
  echo "error: failed to observe an internal IP for ${instance_name} in ${zone}" >&2
  exit 1
}

build_manifest_node_json() {
  local role_id="$1"
  local instance_name="$2"
  local zone="$3"
  local internal_ip="$4"
  local endpoint="$5"
  local endpoint_manifest_uri="$6"
  local bringup_report_uri="$7"
  local runtime_report_uri="$8"
  local node_json profile_json_value
  node_json="$(contract_node_json "${role_id}")"
  profile_json_value="$(profile_json "$(jq -r '.launch_profile_id' <<<"${node_json}")")"
  jq -nc \
    --arg node_id "$(jq -r '.node_id' <<<"${node_json}")" \
    --arg role_id "${role_id}" \
    --arg role_kind "$(jq -r '.role_kind' <<<"${node_json}")" \
    --arg instance_name "${instance_name}" \
    --arg zone "${zone}" \
    --arg subnetwork "$(jq -r '.subnetwork' <<<"${node_json}")" \
    --arg internal_ip "${internal_ip}" \
    --arg endpoint "${endpoint}" \
    --argjson cluster_port "$(jq -r '.cluster_port' <<<"${node_json}")" \
    --arg endpoint_manifest_uri "${endpoint_manifest_uri}" \
    --arg bringup_report_uri "${bringup_report_uri}" \
    --arg runtime_report_uri "${runtime_report_uri}" \
    --arg machine_type "$(jq -r '.machine_type' <<<"${profile_json_value}")" \
    --arg accelerator_type "$(jq -r '.accelerator_type' <<<"${profile_json_value}")" \
    --argjson accelerator_count "$(jq -r '.accelerator_count' <<<"${profile_json_value}")" \
    '{
      node_id: $node_id,
      role_id: $role_id,
      role_kind: $role_kind,
      instance_name: $instance_name,
      zone: $zone,
      subnetwork: $subnetwork,
      internal_ip: (if $internal_ip == "" then null else $internal_ip end),
      endpoint: (if $endpoint == "" then null else $endpoint end),
      cluster_port: $cluster_port,
      endpoint_manifest_uri: $endpoint_manifest_uri,
      bringup_report_uri: $bringup_report_uri,
      runtime_report_uri: $runtime_report_uri,
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: $accelerator_count
    }'
}

build_endpoint_manifest_json() {
  local run_id="$1"
  local role_id="$2"
  local node_id="$3"
  local zone="$4"
  local internal_ip="$5"
  local cluster_port="$6"
  local endpoint="$7"
  jq -nc \
    --arg schema_version "psion.google_two_node_swarm_endpoint_manifest.v1" \
    --arg created_at_utc "$(timestamp_utc)" \
    --arg run_id "${run_id}" \
    --arg node_id "${node_id}" \
    --arg role_id "${role_id}" \
    --arg zone "${zone}" \
    --arg internal_ip "${internal_ip}" \
    --argjson cluster_port "${cluster_port}" \
    --arg endpoint "${endpoint}" \
    --arg source "psion-google-launch-two-node-swarm.sh" \
    '{
      schema_version: $schema_version,
      created_at_utc: $created_at_utc,
      run_id: $run_id,
      node_id: $node_id,
      role_id: $role_id,
      zone: $zone,
      internal_ip: $internal_ip,
      cluster_port: $cluster_port,
      endpoint: $endpoint,
      source: $source
    }'
}

build_cluster_manifest_json() {
  local run_id="$1"
  local cluster_id="$2"
  local selected_pair_id="$3"
  local impairment_profile_id="$4"
  local git_revision="$5"
  local launch_receipt_uri="$6"
  local final_manifest_uri="$7"
  local coordinator_node_json="$8"
  local contributor_node_json="$9"
  local contract_digest cluster_namespace bucket_url training_command_id
  contract_digest="$(jq -r '.contract_digest' "${CONTRACT_FILE}")"
  cluster_namespace="$(jq -r '.cluster_namespace' "${CONTRACT_FILE}")"
  bucket_url="$(jq -r '.bucket_url' "${CONTRACT_FILE}")"
  training_command_id="$(jq -r '.training_command_id' "${CONTRACT_FILE}")"
  jq -nc \
    --arg schema_version "psion.google_two_node_swarm_cluster_manifest.v1" \
    --arg created_at_utc "$(timestamp_utc)" \
    --arg run_id "${run_id}" \
    --arg cluster_id "${cluster_id}" \
    --arg contract_digest "${contract_digest}" \
    --arg cluster_namespace "${cluster_namespace}" \
    --arg project_id "${PROJECT_ID}" \
    --arg region_family "$(jq -r '.region_family' "${CONTRACT_FILE}")" \
    --arg bucket_url "${bucket_url}" \
    --arg run_prefix "${bucket_url}/runs/${run_id}" \
    --arg training_command_id "${training_command_id}" \
    --arg selected_zone_pair_id "${selected_pair_id}" \
    --arg selected_impairment_profile_id "${impairment_profile_id}" \
    --arg git_revision "${git_revision}" \
    --arg launch_receipt_uri "${launch_receipt_uri}" \
    --arg final_manifest_uri "${final_manifest_uri}" \
    --argjson coordinator_node "${coordinator_node_json}" \
    --argjson contributor_node "${contributor_node_json}" \
    '{
      schema_version: $schema_version,
      created_at_utc: $created_at_utc,
      run_id: $run_id,
      cluster_id: $cluster_id,
      contract_digest: $contract_digest,
      cluster_namespace: $cluster_namespace,
      project_id: $project_id,
      region_family: $region_family,
      bucket_url: $bucket_url,
      run_prefix: $run_prefix,
      training_command_id: $training_command_id,
      selected_zone_pair_id: $selected_zone_pair_id,
      selected_impairment_profile_id: $selected_impairment_profile_id,
      git_revision: $git_revision,
      launch_receipt_uri: $launch_receipt_uri,
      final_manifest_uri: $final_manifest_uri,
      nodes: [$coordinator_node, $contributor_node]
    }'
}

build_launch_receipt_json() {
  local run_id="$1"
  local cluster_id="$2"
  local selected_pair_id="$3"
  local impairment_profile_id="$4"
  local manifest_uri="$5"
  local startup_script_uri="$6"
  local preflight_uri="$7"
  local status="$8"
  local detail="$9"
  local coordinator_node_json="${10}"
  local contributor_node_json="${11}"
  local quota_preflight_json="${12}"
  jq -nc \
    --arg schema_version "psion.google_two_node_swarm_launch_receipt.v1" \
    --arg created_at_utc "$(timestamp_utc)" \
    --arg run_id "${run_id}" \
    --arg project_id "${PROJECT_ID}" \
    --arg cluster_id "${cluster_id}" \
    --arg contract_digest "$(jq -r '.contract_digest' "${CONTRACT_FILE}")" \
    --arg training_command_id "$(jq -r '.training_command_id' "${CONTRACT_FILE}")" \
    --arg selected_zone_pair_id "${selected_pair_id}" \
    --arg selected_impairment_profile_id "${impairment_profile_id}" \
    --arg manifest_uri "${manifest_uri}" \
    --arg startup_script_uri "${startup_script_uri}" \
    --arg quota_preflight_uri "${preflight_uri}" \
    --arg git_revision "$(git rev-parse HEAD)" \
    --arg status "${status}" \
    --arg detail "${detail}" \
    --argjson coordinator_node "${coordinator_node_json}" \
    --argjson contributor_node "${contributor_node_json}" \
    --argjson quota_preflight "${quota_preflight_json}" \
    '{
      schema_version: $schema_version,
      created_at_utc: $created_at_utc,
      run_id: $run_id,
      project_id: $project_id,
      cluster_id: $cluster_id,
      contract_digest: $contract_digest,
      training_command_id: $training_command_id,
      selected_zone_pair_id: $selected_zone_pair_id,
      selected_impairment_profile_id: $selected_impairment_profile_id,
      artifact_paths: {
        manifest_uri: $manifest_uri,
        startup_script_uri: $startup_script_uri,
        quota_preflight_uri: $quota_preflight_uri
      },
      quota_preflight: $quota_preflight,
      git_revision: $git_revision,
      status: $status,
      detail: $detail,
      nodes: [$coordinator_node, $contributor_node]
    }'
}

create_instance() {
  local instance_name="$1"
  local zone="$2"
  local role_id="$3"
  local runtime_role="$4"
  local local_endpoint_uri="$5"
  local peer_endpoint_uri="$6"
  local runtime_report_uri="$7"
  local cluster_manifest_uri="$8"
  local low_disk_watermark_gb="$9"
  local profile_json_value identity_json target_tags labels subnetwork
  local machine_type accelerator_type accelerator_count boot_disk_type boot_disk_gb
  local network external_ip os_login provisioning_model maintenance_policy restart_on_failure
  local rust_toolchain metadata_from_file_arg network_interface_arg os_login_metadata
  local restart_on_failure_flag
  profile_json_value="$(profile_json "$(jq -r --arg role_id "${role_id}" '.nodes[] | select(.role_id == $role_id) | .launch_profile_id' "${CONTRACT_FILE}")")"
  identity_json="$(identity_node_json "${role_id}")"
  target_tags="$(target_tags_for_role "${role_id}")"
  labels="$(labels_for_role "${role_id}")"
  subnetwork="$(subnetwork_for_role "${role_id}")"
  network="$(jq -r '.network' "${LAUNCH_FILE}")"
  external_ip="$(jq -r '.external_ip' "${LAUNCH_FILE}")"
  os_login="$(jq -r '.os_login' "${LAUNCH_FILE}")"
  provisioning_model="$(jq -r '.provisioning_model' "${LAUNCH_FILE}")"
  maintenance_policy="$(jq -r '.maintenance_policy' "${LAUNCH_FILE}")"
  restart_on_failure="$(jq -r '.restart_on_failure' "${LAUNCH_FILE}")"
  rust_toolchain="$(jq -r '.startup_policy.rustup_default_toolchain' "${LAUNCH_FILE}")"
  machine_type="$(jq -r '.machine_type' <<<"${profile_json_value}")"
  accelerator_type="$(jq -r '.accelerator_type' <<<"${profile_json_value}")"
  accelerator_count="$(jq -r '.accelerator_count' <<<"${profile_json_value}")"
  boot_disk_type="$(jq -r '.boot_disk_type' <<<"${profile_json_value}")"
  boot_disk_gb="$(jq -r '.boot_disk_gb' <<<"${profile_json_value}")"

  network_interface_arg="network=${network},subnet=${subnetwork},no-address"
  if [[ "${external_ip}" == "true" ]]; then
    network_interface_arg="network=${network},subnet=${subnetwork}"
  fi
  os_login_metadata="FALSE"
  if [[ "${os_login}" == "true" ]]; then
    os_login_metadata="TRUE"
  fi
  restart_on_failure_flag="--no-restart-on-failure"
  if [[ "${restart_on_failure}" == "true" ]]; then
    restart_on_failure_flag="--restart-on-failure"
  fi
  metadata_from_file_arg="startup-script=${STARTUP_SCRIPT},psion-apt-packages=${apt_packages_file},psion-pre-training-command=${pre_training_command_file},psion-training-command=${training_command_file}"

  gcloud compute instances create "${instance_name}" \
    --project="${PROJECT_ID}" \
    --zone="${zone}" \
    --machine-type="${machine_type}" \
    --accelerator="type=${accelerator_type},count=${accelerator_count}" \
    --maintenance-policy="${maintenance_policy}" \
    --provisioning-model="${provisioning_model}" \
    "${restart_on_failure_flag}" \
    --network-interface="${network_interface_arg}" \
    --service-account="$(jq -r '.service_account_email' "${LAUNCH_FILE}")" \
    --scopes="https://www.googleapis.com/auth/cloud-platform" \
    --tags="${target_tags}" \
    --labels="${labels}" \
    --boot-disk-size="${boot_disk_gb}GB" \
    --boot-disk-type="${boot_disk_type}" \
    --image-family="$(jq -r '.image_policy.image_family' "${LAUNCH_FILE}")" \
    --image-project="$(jq -r '.image_policy.image_project' "${LAUNCH_FILE}")" \
    --metadata="enable-oslogin=${os_login_metadata},psion-run-id=${RUN_ID},psion-swarm-role=${runtime_role},psion-role-id=${role_id},psion-node-id=$(jq -r '.node_id' <<<"${profile_json_value}"),psion-bucket-url=$(jq -r '.bucket_url' "${LAUNCH_FILE}"),psion-repo-clone-url=$(jq -r '.repo_clone_url' "${LAUNCH_FILE}"),psion-git-revision=${git_revision},psion-workspace-root=$(jq -r '.startup_policy.workspace_root' "${LAUNCH_FILE}"),psion-rust-toolchain=${rust_toolchain},psion-cluster-manifest-uri=${cluster_manifest_uri},psion-local-endpoint-uri=${local_endpoint_uri},psion-peer-endpoint-uri=${peer_endpoint_uri},psion-runtime-report-uri=${runtime_report_uri},psion-selected-impairment-profile-id=${IMPAIRMENT_PROFILE_ID},psion-endpoint-manifest-timeout-seconds=$(jq -r '.startup_policy.endpoint_manifest_timeout_seconds' "${LAUNCH_FILE}"),psion-endpoint-manifest-poll-interval-seconds=$(jq -r '.startup_policy.endpoint_manifest_poll_interval_seconds' "${LAUNCH_FILE}"),psion-low-disk-watermark-gb=${low_disk_watermark_gb}" \
    --metadata-from-file="${metadata_from_file_arg}" \
    --quiet >/dev/null
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

contract_digest="$(jq -r '.contract_digest' "${CONTRACT_FILE}")"
launch_contract_digest="$(jq -r '.contract_digest' "${LAUNCH_FILE}")"
if [[ "${contract_digest}" != "${launch_contract_digest}" ]]; then
  echo "error: launch profile contract digest ${launch_contract_digest} does not match contract digest ${contract_digest}" >&2
  exit 1
fi

if ! jq -e --arg profile_id "${IMPAIRMENT_PROFILE_ID}" '.admitted_impairment_profile_ids[] | select(. == $profile_id)' "${CONTRACT_FILE}" >/dev/null; then
  echo "error: unsupported impairment profile ${IMPAIRMENT_PROFILE_ID}" >&2
  exit 1
fi

quota_preflight_json="$(PROJECT_ID="${PROJECT_ID}" PAIR_ID="${PAIR_ID}" bash "${QUOTA_PREFLIGHT}")"
if [[ "$(jq -r '.result' <<<"${quota_preflight_json}")" != "ready" ]]; then
  echo "error: quota preflight did not return ready" >&2
  exit 1
fi
selected_pair_json="$(jq -c '.selected_pair' <<<"${quota_preflight_json}")"
selected_pair_id="$(jq -r '.pair_id' <<<"${selected_pair_json}")"
coordinator_zone="$(jq -r '.coordinator_zone' <<<"${selected_pair_json}")"
contributor_zone="$(jq -r '.contributor_zone' <<<"${selected_pair_json}")"
if ! jq -e --arg pair_id "${selected_pair_id}" '.zone_pair_fallback_order[] | select(. == $pair_id)' "${LAUNCH_FILE}" >/dev/null; then
  echo "error: selected pair ${selected_pair_id} is not admitted by the launch authority" >&2
  exit 1
fi

timestamp_tag="$(date -u '+%Y%m%dt%H%M%Sz' | tr '[:upper:]' '[:lower:]')"
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="psion-google-swarm-${timestamp_tag}"
fi
if [[ -z "${COORDINATOR_INSTANCE_NAME}" ]]; then
  COORDINATOR_INSTANCE_NAME="$(make_instance_name "${RUN_ID}" "coord")"
fi
if [[ -z "${CONTRIBUTOR_INSTANCE_NAME}" ]]; then
  CONTRIBUTOR_INSTANCE_NAME="$(make_instance_name "${RUN_ID}" "contrib")"
fi

run_prefix="$(jq -r '.bucket_url' "${LAUNCH_FILE}")/runs/${RUN_ID}"
manifest_uri="${run_prefix}/$(jq -r '.artifact_paths.cluster_manifest_object' "${LAUNCH_FILE}")"
launch_receipt_uri="${run_prefix}/$(jq -r '.artifact_paths.launch_receipt_object' "${LAUNCH_FILE}")"
startup_script_uri="${run_prefix}/$(jq -r '.artifact_paths.startup_script_object' "${LAUNCH_FILE}")"
preflight_uri="${run_prefix}/$(jq -r '.artifact_paths.quota_preflight_object' "${LAUNCH_FILE}")"
coordinator_endpoint_manifest_uri="${run_prefix}/$(jq -r '.artifact_paths.coordinator_endpoint_manifest_object' "${LAUNCH_FILE}")"
contributor_endpoint_manifest_uri="${run_prefix}/$(jq -r '.artifact_paths.contributor_endpoint_manifest_object' "${LAUNCH_FILE}")"
coordinator_runtime_report_uri="${run_prefix}/$(jq -r '.artifact_paths.coordinator_runtime_report_object' "${LAUNCH_FILE}")"
contributor_runtime_report_uri="${run_prefix}/$(jq -r '.artifact_paths.contributor_runtime_report_object' "${LAUNCH_FILE}")"
final_manifest_uri="${run_prefix}/$(jq -r '.artifact_paths.final_manifest_object' "${LAUNCH_FILE}")"

git_revision="$(git rev-parse HEAD)"
startup_script_sha256="$(compute_sha256 "${STARTUP_SCRIPT}")"

apt_packages_file="${tmpdir}/apt_packages.txt"
pre_training_command_file="${tmpdir}/pre_training_command.sh"
training_command_file="${tmpdir}/training_command.sh"
quota_preflight_file="${tmpdir}/psion_google_two_node_swarm_quota_preflight.json"
cluster_manifest_file="${tmpdir}/psion_google_two_node_swarm_cluster_manifest.json"
launch_receipt_file="${tmpdir}/psion_google_two_node_swarm_launch_receipt.json"
coordinator_endpoint_manifest_file="${tmpdir}/coordinator_endpoint_manifest.json"
contributor_endpoint_manifest_file="${tmpdir}/contributor_endpoint_manifest.json"

jq -r '.startup_policy.package_install | join(" ")' "${LAUNCH_FILE}" > "${apt_packages_file}"
printf '%s\n' "$(jq -r '.startup_policy.pre_training_command' "${LAUNCH_FILE}")" > "${pre_training_command_file}"
printf '%s\n' "$(jq -r '.startup_policy.training_command' "${LAUNCH_FILE}")" > "${training_command_file}"
printf '%s\n' "${quota_preflight_json}" > "${quota_preflight_file}"

cluster_namespace="$(jq -r '.cluster_namespace' "${CONTRACT_FILE}")"
cluster_id="$(compute_cluster_id "${RUN_ID}" "${contract_digest}" "${cluster_namespace}")"

coordinator_role_id="psion.google_swarm.coordinator_validator_aggregator_contributor"
contributor_role_id="psion.google_swarm.contributor"

coordinator_profile_json="$(profile_json "g2_l4_two_node_swarm_coordinator")"
contributor_profile_json="$(profile_json "g2_l4_two_node_swarm_contributor")"

coordinator_bringup_report_uri="${run_prefix}/$(jq -r '.bucket_authority.coordinator_bringup_report_object' "${CONTRACT_FILE}")"
contributor_bringup_report_uri="${run_prefix}/$(jq -r '.bucket_authority.contributor_bringup_report_object' "${CONTRACT_FILE}")"

gcloud storage cp --quiet "${STARTUP_SCRIPT}" "${startup_script_uri}" >/dev/null
gcloud storage cp --quiet "${quota_preflight_file}" "${preflight_uri}" >/dev/null
wait_for_object "${startup_script_uri}"
wait_for_object "${preflight_uri}"

if [[ "${MANIFEST_ONLY}" == "true" ]]; then
  coordinator_node_json="$(build_manifest_node_json \
    "${coordinator_role_id}" \
    "${COORDINATOR_INSTANCE_NAME}" \
    "${coordinator_zone}" \
    "" \
    "" \
    "${coordinator_endpoint_manifest_uri}" \
    "${coordinator_bringup_report_uri}" \
    "${coordinator_runtime_report_uri}")"
  contributor_node_json="$(build_manifest_node_json \
    "${contributor_role_id}" \
    "${CONTRIBUTOR_INSTANCE_NAME}" \
    "${contributor_zone}" \
    "" \
    "" \
    "${contributor_endpoint_manifest_uri}" \
    "${contributor_bringup_report_uri}" \
    "${contributor_runtime_report_uri}")"
  build_cluster_manifest_json \
    "${RUN_ID}" \
    "${cluster_id}" \
    "${selected_pair_id}" \
    "${IMPAIRMENT_PROFILE_ID}" \
    "${git_revision}" \
    "${launch_receipt_uri}" \
    "${final_manifest_uri}" \
    "${coordinator_node_json}" \
    "${contributor_node_json}" > "${cluster_manifest_file}"
  build_launch_receipt_json \
    "${RUN_ID}" \
    "${cluster_id}" \
    "${selected_pair_id}" \
    "${IMPAIRMENT_PROFILE_ID}" \
    "${manifest_uri}" \
    "${startup_script_uri}" \
    "${preflight_uri}" \
    "manifest_only" \
    "The dual-node launch bundle was materialized without creating either Google VM, so the manifest keeps node roles, zones, subnetworks, URIs, and training command truth explicit while internal endpoints remain pending allocation." \
    "${coordinator_node_json}" \
    "${contributor_node_json}" \
    "${quota_preflight_json}" > "${launch_receipt_file}"
  gcloud storage cp --quiet "${cluster_manifest_file}" "${manifest_uri}" >/dev/null
  gcloud storage cp --quiet "${launch_receipt_file}" "${launch_receipt_uri}" >/dev/null
  wait_for_object "${manifest_uri}"
  wait_for_object "${launch_receipt_uri}"
  echo "manifest-only launch bundle uploaded:"
  cat "${cluster_manifest_file}"
  exit 0
fi

launch_complete=false
cleanup_on_failure() {
  if [[ "${launch_complete}" == "true" ]]; then
    return 0
  fi
  for entry in \
    "${COORDINATOR_INSTANCE_NAME}:${coordinator_zone}" \
    "${CONTRIBUTOR_INSTANCE_NAME}:${contributor_zone}"; do
    instance_name="${entry%%:*}"
    zone="${entry##*:}"
    if gcloud compute instances describe "${instance_name}" --project="${PROJECT_ID}" --zone="${zone}" >/dev/null 2>&1; then
      gcloud compute instances delete "${instance_name}" \
        --project="${PROJECT_ID}" \
        --zone="${zone}" \
        --quiet >/dev/null || true
    fi
  done
}
trap 'cleanup_on_failure; rm -rf "${tmpdir}"' EXIT

create_instance \
  "${COORDINATOR_INSTANCE_NAME}" \
  "${coordinator_zone}" \
  "${coordinator_role_id}" \
  "coordinator" \
  "${coordinator_endpoint_manifest_uri}" \
  "${contributor_endpoint_manifest_uri}" \
  "${coordinator_runtime_report_uri}" \
  "${manifest_uri}" \
  "$(jq -r '.low_disk_watermark_gb' <<<"${coordinator_profile_json}")"

create_instance \
  "${CONTRIBUTOR_INSTANCE_NAME}" \
  "${contributor_zone}" \
  "${contributor_role_id}" \
  "contributor" \
  "${contributor_endpoint_manifest_uri}" \
  "${coordinator_endpoint_manifest_uri}" \
  "${contributor_runtime_report_uri}" \
  "${manifest_uri}" \
  "$(jq -r '.low_disk_watermark_gb' <<<"${contributor_profile_json}")"

coordinator_internal_ip="$(wait_for_internal_ip "${COORDINATOR_INSTANCE_NAME}" "${coordinator_zone}")"
contributor_internal_ip="$(wait_for_internal_ip "${CONTRIBUTOR_INSTANCE_NAME}" "${contributor_zone}")"
coordinator_endpoint="${coordinator_internal_ip}:$(jq -r '.cluster_port' <<<"${coordinator_profile_json}")"
contributor_endpoint="${contributor_internal_ip}:$(jq -r '.cluster_port' <<<"${contributor_profile_json}")"

coordinator_node_json="$(build_manifest_node_json \
  "${coordinator_role_id}" \
  "${COORDINATOR_INSTANCE_NAME}" \
  "${coordinator_zone}" \
  "${coordinator_internal_ip}" \
  "${coordinator_endpoint}" \
  "${coordinator_endpoint_manifest_uri}" \
  "${coordinator_bringup_report_uri}" \
  "${coordinator_runtime_report_uri}")"
contributor_node_json="$(build_manifest_node_json \
  "${contributor_role_id}" \
  "${CONTRIBUTOR_INSTANCE_NAME}" \
  "${contributor_zone}" \
  "${contributor_internal_ip}" \
  "${contributor_endpoint}" \
  "${contributor_endpoint_manifest_uri}" \
  "${contributor_bringup_report_uri}" \
  "${contributor_runtime_report_uri}")"

build_endpoint_manifest_json \
  "${RUN_ID}" \
  "${coordinator_role_id}" \
  "$(jq -r '.node_id' <<<"${coordinator_profile_json}")" \
  "${coordinator_zone}" \
  "${coordinator_internal_ip}" \
  "$(jq -r '.cluster_port' <<<"${coordinator_profile_json}")" \
  "${coordinator_endpoint}" > "${coordinator_endpoint_manifest_file}"
build_endpoint_manifest_json \
  "${RUN_ID}" \
  "${contributor_role_id}" \
  "$(jq -r '.node_id' <<<"${contributor_profile_json}")" \
  "${contributor_zone}" \
  "${contributor_internal_ip}" \
  "$(jq -r '.cluster_port' <<<"${contributor_profile_json}")" \
  "${contributor_endpoint}" > "${contributor_endpoint_manifest_file}"

build_cluster_manifest_json \
  "${RUN_ID}" \
  "${cluster_id}" \
  "${selected_pair_id}" \
  "${IMPAIRMENT_PROFILE_ID}" \
  "${git_revision}" \
  "${launch_receipt_uri}" \
  "${final_manifest_uri}" \
  "${coordinator_node_json}" \
  "${contributor_node_json}" > "${cluster_manifest_file}"
build_launch_receipt_json \
  "${RUN_ID}" \
  "${cluster_id}" \
  "${selected_pair_id}" \
  "${IMPAIRMENT_PROFILE_ID}" \
  "${manifest_uri}" \
  "${startup_script_uri}" \
  "${preflight_uri}" \
  "instances_created" \
  "The exact two-node Google configured-peer swarm shape was launched, both nodes now expose explicit private endpoints, and the role-aware startup script can begin the bounded adapter-delta runtime as soon as the uploaded manifests become visible." \
  "${coordinator_node_json}" \
  "${contributor_node_json}" \
  "${quota_preflight_json}" > "${launch_receipt_file}"

gcloud storage cp --quiet "${coordinator_endpoint_manifest_file}" "${coordinator_endpoint_manifest_uri}" >/dev/null
gcloud storage cp --quiet "${contributor_endpoint_manifest_file}" "${contributor_endpoint_manifest_uri}" >/dev/null
gcloud storage cp --quiet "${cluster_manifest_file}" "${manifest_uri}" >/dev/null
gcloud storage cp --quiet "${launch_receipt_file}" "${launch_receipt_uri}" >/dev/null
wait_for_object "${coordinator_endpoint_manifest_uri}"
wait_for_object "${contributor_endpoint_manifest_uri}"
wait_for_object "${manifest_uri}"
wait_for_object "${launch_receipt_uri}"

launch_complete=true

echo "launched Google two-node swarm run ${RUN_ID}"
jq -n \
  --arg run_id "${RUN_ID}" \
  --arg cluster_id "${cluster_id}" \
  --arg pair_id "${selected_pair_id}" \
  --arg impairment_profile_id "${IMPAIRMENT_PROFILE_ID}" \
  --arg manifest_uri "${manifest_uri}" \
  --arg launch_receipt_uri "${launch_receipt_uri}" \
  --arg coordinator_instance_name "${COORDINATOR_INSTANCE_NAME}" \
  --arg coordinator_zone "${coordinator_zone}" \
  --arg coordinator_endpoint "${coordinator_endpoint}" \
  --arg contributor_instance_name "${CONTRIBUTOR_INSTANCE_NAME}" \
  --arg contributor_zone "${contributor_zone}" \
  --arg contributor_endpoint "${contributor_endpoint}" \
  '{
    run_id: $run_id,
    cluster_id: $cluster_id,
    pair_id: $pair_id,
    impairment_profile_id: $impairment_profile_id,
    manifest_uri: $manifest_uri,
    launch_receipt_uri: $launch_receipt_uri,
    nodes: [
      {
        runtime_role: "coordinator",
        instance_name: $coordinator_instance_name,
        zone: $coordinator_zone,
        endpoint: $coordinator_endpoint
      },
      {
        runtime_role: "contributor",
        instance_name: $contributor_instance_name,
        zone: $contributor_zone,
        endpoint: $contributor_endpoint
      }
    ]
  }'

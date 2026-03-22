#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
PROFILE_ID="${PROFILE_ID:-g2_l4_single_node}"
ZONE="${ZONE:-}"
RUN_ID="${RUN_ID:-}"
INSTANCE_NAME="${INSTANCE_NAME:-}"
MANIFEST_ONLY=false

usage() {
  cat <<'EOF'
Usage: psion-google-launch-single-node.sh [options]

Options:
  --profile <profile_id>      Launch profile from the committed authority file.
  --zone <zone>               Force one zone instead of walking the fallback order.
  --run-id <run_id>           Override the generated run id.
  --instance-name <name>      Override the generated instance name.
  --manifest-only             Upload the launch manifest and startup snapshot without creating a VM.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE_ID="$2"
      shift 2
      ;;
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --instance-name)
      INSTANCE_NAME="$2"
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
      usage
      exit 1
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
LAUNCH_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_single_node_launch_profiles_v1.json"
STARTUP_SCRIPT="${REPO_ROOT}/scripts/psion-google-single-node-startup.sh"
QUOTA_PREFLIGHT="${REPO_ROOT}/scripts/psion-google-quota-preflight.sh"

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

sanitize_label() {
  tr '[:upper:]_' '[:lower:]-' <<<"$1" | tr -cd 'a-z0-9-'
}

wait_for_object() {
  local object_path="$1"
  local attempt
  for attempt in 1 2 3 4 5; do
    if gcloud storage ls "${object_path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: object ${object_path} did not become visible" >&2
  exit 1
}

profile_json="$(
  jq -c --arg profile_id "${PROFILE_ID}" '.profiles[] | select(.profile_id == $profile_id)' "${LAUNCH_FILE}"
)"

if [[ -z "${profile_json}" ]]; then
  echo "error: unknown profile ${PROFILE_ID}" >&2
  exit 1
fi

bucket_url="$(jq -r '.bucket_url' "${LAUNCH_FILE}")"
repo_clone_url="$(jq -r '.repo_clone_url' "${LAUNCH_FILE}")"
service_account_email="$(jq -r '.service_account_email' "${LAUNCH_FILE}")"
network="$(jq -r '.network' "${LAUNCH_FILE}")"
subnetwork="$(jq -r '.subnetwork' "${LAUNCH_FILE}")"
target_tags="$(jq -r '.target_tags | join(",")' "${LAUNCH_FILE}")"
image_project="$(jq -r '.image_policy.image_project' "${LAUNCH_FILE}")"
image_family="$(jq -r '.image_policy.image_family' "${LAUNCH_FILE}")"
driver_posture="$(jq -r '.image_policy.driver_posture' "${LAUNCH_FILE}")"
workspace_root="$(jq -r '.startup_policy.workspace_root' "${LAUNCH_FILE}")"
training_command="$(jq -r '.startup_policy.training_command' "${LAUNCH_FILE}")"
rust_toolchain="$(jq -r '.startup_policy.rustup_default_toolchain' "${LAUNCH_FILE}")"
apt_packages="$(jq -r '.startup_policy.package_install | join(" ")' "${LAUNCH_FILE}")"
final_manifest_object="$(jq -r '.teardown_policy.final_manifest_object' "${LAUNCH_FILE}")"

profile_label="$(jq -r '.profile_label' <<<"${profile_json}")"
machine_type="$(jq -r '.machine_type' <<<"${profile_json}")"
accelerator_type="$(jq -r '.accelerator_type' <<<"${profile_json}")"
accelerator_count="$(jq -r '.accelerator_count' <<<"${profile_json}")"
boot_disk_type="$(jq -r '.boot_disk_type' <<<"${profile_json}")"
boot_disk_gb="$(jq -r '.boot_disk_gb' <<<"${profile_json}")"
low_disk_watermark_gb="$(jq -r '.low_disk_watermark_gb' <<<"${profile_json}")"
declared_run_cost_ceiling_usd="$(jq -r '.declared_run_cost_ceiling_usd' <<<"${profile_json}")"

timestamp_tag="$(date -u '+%Y%m%dt%H%M%Sz' | tr '[:upper:]' '[:lower:]')"
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="psion-${profile_label}-${timestamp_tag}"
fi
if [[ -z "${INSTANCE_NAME}" ]]; then
  INSTANCE_NAME="${RUN_ID}"
fi

selected_zone=""
zone_selection_reason=""
preflight_report_file=""

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

if [[ -n "${ZONE}" ]]; then
  candidate_zones=("${ZONE}")
else
  candidate_zones=()
  while IFS= read -r candidate_zone; do
    candidate_zones+=("${candidate_zone}")
  done < <(jq -r '.zone_fallback_order[]' <<<"${profile_json}")
fi

for candidate_zone in "${candidate_zones[@]}"; do
  preflight_candidate="${tmpdir}/${PROFILE_ID}-${candidate_zone}.json"
  if PROFILE_ID="${PROFILE_ID}" ZONE="${candidate_zone}" bash "${QUOTA_PREFLIGHT}" > "${preflight_candidate}" 2>/dev/null; then
    selected_zone="${candidate_zone}"
    zone_selection_reason="quota_preflight_ready"
    preflight_report_file="${preflight_candidate}"
    break
  fi
done

if [[ -z "${selected_zone}" ]]; then
  echo "error: no zone passed quota preflight for ${PROFILE_ID}" >&2
  exit 1
fi

image_json="$(
  gcloud compute images describe-from-family "${image_family}" \
    --project="${image_project}" \
    --format=json
)"
image_name="$(jq -r '.name' <<<"${image_json}")"
image_self_link="$(jq -r '.selfLink' <<<"${image_json}")"
image_id="$(jq -r '.id' <<<"${image_json}")"
image_creation_timestamp="$(jq -r '.creationTimestamp' <<<"${image_json}")"

git_revision="$(git rev-parse HEAD)"
startup_script_sha256="$(compute_sha256 "${STARTUP_SCRIPT}")"
run_prefix="${bucket_url}/runs/${RUN_ID}"
launch_prefix="${run_prefix}/launch"
manifest_uri="${launch_prefix}/psion_google_single_node_launch_manifest.json"
startup_script_uri="${launch_prefix}/psion-google-single-node-startup.sh"
preflight_uri="${launch_prefix}/psion_google_quota_preflight.json"
final_manifest_uri="${run_prefix}/${final_manifest_object}"

manifest_file="${tmpdir}/psion_google_single_node_launch_manifest.json"
training_command_file="${tmpdir}/training_command.sh"
apt_packages_file="${tmpdir}/apt_packages.txt"
printf '%s\n' "${training_command}" > "${training_command_file}"
printf '%s\n' "${apt_packages}" > "${apt_packages_file}"

jq -n \
  --arg schema_version "psion.google_single_node_launch_manifest.v1" \
  --arg created_at_utc "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  --arg run_id "${RUN_ID}" \
  --arg instance_name "${INSTANCE_NAME}" \
  --arg project_id "${PROJECT_ID}" \
  --arg profile_id "${PROFILE_ID}" \
  --arg profile_label "${profile_label}" \
  --arg zone "${selected_zone}" \
  --arg zone_selection_reason "${zone_selection_reason}" \
  --arg machine_type "${machine_type}" \
  --arg accelerator_type "${accelerator_type}" \
  --argjson accelerator_count "${accelerator_count}" \
  --arg service_account_email "${service_account_email}" \
  --arg network "${network}" \
  --arg subnetwork "${subnetwork}" \
  --arg target_tags "${target_tags}" \
  --arg bucket_url "${bucket_url}" \
  --arg manifest_uri "${manifest_uri}" \
  --arg startup_script_uri "${startup_script_uri}" \
  --arg preflight_uri "${preflight_uri}" \
  --arg final_manifest_uri "${final_manifest_uri}" \
  --arg image_project "${image_project}" \
  --arg image_family "${image_family}" \
  --arg image_name "${image_name}" \
  --arg image_self_link "${image_self_link}" \
  --arg image_id "${image_id}" \
  --arg image_creation_timestamp "${image_creation_timestamp}" \
  --arg driver_posture "${driver_posture}" \
  --arg startup_script_path "scripts/psion-google-single-node-startup.sh" \
  --arg startup_script_sha256 "${startup_script_sha256}" \
  --arg repo_clone_url "${repo_clone_url}" \
  --arg git_revision "${git_revision}" \
  --arg training_command "${training_command}" \
  --arg workspace_root "${workspace_root}" \
  --arg output_subdir "output" \
  --arg log_subdir "logs" \
  --arg scratch_subdir "scratch" \
  --arg repo_subdir "repo" \
  --arg rust_toolchain "${rust_toolchain}" \
  --argjson package_install "$(jq '.startup_policy.package_install' "${LAUNCH_FILE}")" \
  --arg boot_disk_type "${boot_disk_type}" \
  --argjson boot_disk_gb "${boot_disk_gb}" \
  --argjson low_disk_watermark_gb "${low_disk_watermark_gb}" \
  --argjson declared_run_cost_ceiling_usd "${declared_run_cost_ceiling_usd}" \
  --arg provisioning_model "$(jq -r '.provisioning_model' "${LAUNCH_FILE}")" \
  --arg maintenance_policy "$(jq -r '.maintenance_policy' "${LAUNCH_FILE}")" \
  --argjson restart_on_failure "$(jq -r '.restart_on_failure' "${LAUNCH_FILE}")" \
  --argjson external_ip "$(jq -r '.external_ip' "${LAUNCH_FILE}")" \
  --argjson os_login "$(jq -r '.os_login' "${LAUNCH_FILE}")" \
  --argjson quota_preflight "$(cat "${preflight_report_file}")" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    instance_name: $instance_name,
    project_id: $project_id,
    profile_id: $profile_id,
    profile_label: $profile_label,
    zone: $zone,
    zone_selection_reason: $zone_selection_reason,
    quota_preflight: $quota_preflight,
    machine: {
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: $accelerator_count,
      provisioning_model: $provisioning_model,
      maintenance_policy: $maintenance_policy,
      restart_on_failure: $restart_on_failure
    },
    network: {
      network: $network,
      subnetwork: $subnetwork,
      target_tags: ($target_tags | split(",")),
      external_ip: $external_ip,
      os_login: $os_login
    },
    image: {
      image_project: $image_project,
      image_family: $image_family,
      image_name: $image_name,
      image_self_link: $image_self_link,
      image_id: $image_id,
      image_creation_timestamp: $image_creation_timestamp,
      driver_posture: $driver_posture
    },
    storage: {
      boot_disk_type: $boot_disk_type,
      boot_disk_gb: $boot_disk_gb,
      low_disk_watermark_gb: $low_disk_watermark_gb,
      bucket_url: $bucket_url,
      run_prefix: ($bucket_url + "/runs/" + $run_id),
      final_manifest_uri: $final_manifest_uri
    },
    repo: {
      clone_url: $repo_clone_url,
      git_revision: $git_revision
    },
    startup: {
      script_path: $startup_script_path,
      script_sha256: $startup_script_sha256,
      workspace_root: $workspace_root,
      output_subdir: $output_subdir,
      log_subdir: $log_subdir,
      scratch_subdir: $scratch_subdir,
      repo_subdir: $repo_subdir,
      rust_toolchain: $rust_toolchain,
      package_install: $package_install
    },
    training: {
      command: $training_command,
      declared_run_cost_ceiling_usd: $declared_run_cost_ceiling_usd
    },
    artifact_paths: {
      manifest_uri: $manifest_uri,
      startup_script_uri: $startup_script_uri,
      quota_preflight_uri: $preflight_uri
    }
  }' > "${manifest_file}"

gcloud storage cp --quiet "${manifest_file}" "${manifest_uri}" >/dev/null
gcloud storage cp --quiet "${STARTUP_SCRIPT}" "${startup_script_uri}" >/dev/null
gcloud storage cp --quiet "${preflight_report_file}" "${preflight_uri}" >/dev/null
wait_for_object "${manifest_uri}"
wait_for_object "${startup_script_uri}"
wait_for_object "${preflight_uri}"

if [[ "${MANIFEST_ONLY}" == "true" ]]; then
  echo "launch manifest uploaded:"
  cat "${manifest_file}"
  exit 0
fi

gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${selected_zone}" \
  --machine-type="${machine_type}" \
  --accelerator="type=${accelerator_type},count=${accelerator_count}" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --no-restart-on-failure \
  --network-interface="network=${network},subnet=${subnetwork},no-address" \
  --service-account="${service_account_email}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --tags="${target_tags}" \
  --labels="lane=psion,purpose=train,track=single-node,profile=$(sanitize_label "${profile_label}")" \
  --boot-disk-size="${boot_disk_gb}GB" \
  --boot-disk-type="${boot_disk_type}" \
  --image="${image_name}" \
  --image-project="${image_project}" \
  --metadata="enable-oslogin=TRUE,psion-run-id=${RUN_ID},psion-bucket-url=${bucket_url},psion-repo-clone-url=${repo_clone_url},psion-git-revision=${git_revision},psion-workspace-root=${workspace_root},psion-output-subdir=output,psion-log-subdir=logs,psion-scratch-subdir=scratch,psion-repo-subdir=repo,psion-low-disk-watermark-gb=${low_disk_watermark_gb},psion-rust-toolchain=${rust_toolchain}" \
  --metadata-from-file="startup-script=${STARTUP_SCRIPT},psion-training-command=${training_command_file},psion-apt-packages=${apt_packages_file}" >/dev/null

echo "instance launched:"
gcloud compute instances describe "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${selected_zone}" \
  --format=json | jq '{name, zone: (.zone | split("/") | last), machineType: (.machineType | split("/") | last), guestAccelerators, serviceAccounts, labels, status}'

echo "launch manifest uri: ${manifest_uri}"
echo "iap ssh: gcloud compute ssh ${INSTANCE_NAME} --project=${PROJECT_ID} --zone=${selected_zone} --tunnel-through-iap"

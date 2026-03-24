#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
PROFILE_ID="${PROFILE_ID:-g2_l4_single_node_accelerated}"
ZONE="${ZONE:-}"
RUN_ID="${RUN_ID:-}"
INSTANCE_NAME="${INSTANCE_NAME:-}"
INPUT_PACKAGE_DESCRIPTOR_URI_OVERRIDE="${INPUT_PACKAGE_DESCRIPTOR_URI_OVERRIDE:-}"
PRE_TRAINING_COMMAND_OVERRIDE="${PRE_TRAINING_COMMAND_OVERRIDE:-}"
TRAINING_COMMAND_OVERRIDE="${TRAINING_COMMAND_OVERRIDE:-}"
POST_TRAINING_ARCHIVE_COMMAND_OVERRIDE="${POST_TRAINING_ARCHIVE_COMMAND_OVERRIDE:-}"
POST_TRAINING_RESTORE_COMMAND_OVERRIDE="${POST_TRAINING_RESTORE_COMMAND_OVERRIDE:-}"
MANIFEST_ONLY=false

usage() {
  cat <<'EOF'
Usage: psion-google-launch-single-node.sh [options]

Options:
  --profile <profile_id>      Launch profile from the committed authority file.
  --zone <zone>               Force one zone instead of walking the fallback order.
  --run-id <run_id>           Override the generated run id.
  --instance-name <name>      Override the generated instance name.
  --input-package-descriptor-uri <uri>
                              Override the committed input-package descriptor.
  --pre-training-command <cmd>
                              Override the repo-owned pre-training command.
  --training-command <cmd>    Override the repo-owned training command.
  --post-training-archive-command <cmd>
                              Override the post-training archive command.
  --post-training-restore-command <cmd>
                              Override the post-training restore command.
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
    --input-package-descriptor-uri)
      INPUT_PACKAGE_DESCRIPTOR_URI_OVERRIDE="$2"
      shift 2
      ;;
    --pre-training-command)
      PRE_TRAINING_COMMAND_OVERRIDE="$2"
      shift 2
      ;;
    --training-command)
      TRAINING_COMMAND_OVERRIDE="$2"
      shift 2
      ;;
    --post-training-archive-command)
      POST_TRAINING_ARCHIVE_COMMAND_OVERRIDE="$2"
      shift 2
      ;;
    --post-training-restore-command)
      POST_TRAINING_RESTORE_COMMAND_OVERRIDE="$2"
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
LAUNCH_FILE="${LAUNCH_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_single_node_launch_profiles_v1.json}"
OBSERVABILITY_FILE="${OBSERVABILITY_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_host_observability_policy_v1.json}"
STARTUP_SCRIPT="${STARTUP_SCRIPT:-${REPO_ROOT}/scripts/psion-google-single-node-startup.sh}"
QUOTA_PREFLIGHT="${QUOTA_PREFLIGHT:-${REPO_ROOT}/scripts/psion-google-quota-preflight.sh}"

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

write_launch_failure_manifest() {
  local failure_code="$1"
  local failure_detail="$2"
  local failure_log_file="$3"
  local created_at_utc manifest_of_manifests_uri launch_failure_log_uri
  local manifest_of_manifests_file manifest_of_manifests_sha256 final_manifest_file

  created_at_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  launch_failure_log_uri="${run_prefix}/$(jq -r '.artifact_paths.final_prefix' "${OBSERVABILITY_FILE}")/$(jq -r '.launch_failure_log_name' "${OBSERVABILITY_FILE}")"
  manifest_of_manifests_uri="${run_prefix}/$(jq -r '.artifact_paths.final_prefix' "${OBSERVABILITY_FILE}")/$(jq -r '.manifest_of_manifests_name' "${OBSERVABILITY_FILE}")"
  manifest_of_manifests_file="${tmpdir}/$(jq -r '.manifest_of_manifests_name' "${OBSERVABILITY_FILE}")"
  final_manifest_file="${tmpdir}/$(jq -r '.final_manifest_name' "${OBSERVABILITY_FILE}")"

  gcloud storage cp --quiet "${failure_log_file}" "${launch_failure_log_uri}" >/dev/null
  wait_for_object "${launch_failure_log_uri}"

  jq -n \
    --arg schema_version "psion.google_run_manifest_of_manifests.v1" \
    --arg created_at_utc "${created_at_utc}" \
    --arg run_id "${RUN_ID}" \
    --arg manifest_uri "${manifest_uri}" \
    --arg manifest_sha256 "$(compute_sha256 "${manifest_file}")" \
    --arg manifest_bytes "$(wc -c < "${manifest_file}" | tr -d ' ')" \
    --arg preflight_uri "${preflight_uri}" \
    --arg preflight_sha256 "$(compute_sha256 "${preflight_report_file}")" \
    --arg preflight_bytes "$(wc -c < "${preflight_report_file}" | tr -d ' ')" \
    '{
      schema_version: $schema_version,
      created_at_utc: $created_at_utc,
      run_id: $run_id,
      manifests: [
        {
          artifact_kind: "launch_manifest",
          evidence_role: "manifest",
          remote_uri: $manifest_uri,
          sha256: $manifest_sha256,
          byte_length: ($manifest_bytes | tonumber),
          source_mode: "local_upload"
        },
        {
          artifact_kind: "quota_preflight",
          evidence_role: "manifest",
          remote_uri: $preflight_uri,
          sha256: $preflight_sha256,
          byte_length: ($preflight_bytes | tonumber),
          source_mode: "local_upload"
        }
      ]
    }' > "${manifest_of_manifests_file}"
  manifest_of_manifests_sha256="$(compute_sha256 "${manifest_of_manifests_file}")"
  gcloud storage cp --quiet "${manifest_of_manifests_file}" "${manifest_of_manifests_uri}" >/dev/null
  wait_for_object "${manifest_of_manifests_uri}"

  jq -n \
    --arg schema_version "psion.google_run_final_manifest.v1" \
    --arg created_at_utc "${created_at_utc}" \
    --arg run_id "${RUN_ID}" \
    --arg project_id "${PROJECT_ID}" \
    --arg zone "${selected_zone}" \
    --arg instance_name "${INSTANCE_NAME}" \
    --arg profile_id "${PROFILE_ID}" \
    --arg profile_label "${profile_label}" \
    --arg machine_type "${machine_type}" \
    --arg accelerator_type "${accelerator_type}" \
    --argjson accelerator_count "${accelerator_count}" \
    --arg result_classification "${failure_code}" \
    --arg failure_detail "${failure_detail}" \
    --arg launch_manifest_uri "${manifest_uri}" \
    --arg startup_script_uri "${startup_script_uri}" \
    --arg quota_preflight_uri "${preflight_uri}" \
    --arg launch_failure_log_uri "${launch_failure_log_uri}" \
    --arg manifest_of_manifests_uri "${manifest_of_manifests_uri}" \
    --arg manifest_of_manifests_sha256 "${manifest_of_manifests_sha256}" \
    --arg launch_created_at_utc "$(jq -r '.created_at_utc' "${manifest_file}")" \
    --arg manifest_sha256 "$(compute_sha256 "${manifest_file}")" \
    --arg manifest_bytes "$(wc -c < "${manifest_file}" | tr -d ' ')" \
    --arg startup_script_sha256 "${startup_script_sha256}" \
    --arg startup_script_bytes "$(wc -c < "${STARTUP_SCRIPT}" | tr -d ' ')" \
    --arg preflight_sha256 "$(compute_sha256 "${preflight_report_file}")" \
    --arg preflight_bytes "$(wc -c < "${preflight_report_file}" | tr -d ' ')" \
    --arg launch_failure_log_sha256 "$(compute_sha256 "${failure_log_file}")" \
    --arg launch_failure_log_bytes "$(wc -c < "${failure_log_file}" | tr -d ' ')" \
    '{
      schema_version: $schema_version,
      created_at_utc: $created_at_utc,
      run_id: $run_id,
      project_id: $project_id,
      topology: {
        zone: $zone,
        instance_name: $instance_name,
        profile_id: $profile_id,
        profile_label: $profile_label,
        machine_type: $machine_type,
        accelerator_type: $accelerator_type,
        accelerator_count: $accelerator_count
      },
      result_classification: $result_classification,
      failure_detail: $failure_detail,
      launch_artifacts: {
        launch_manifest_uri: $launch_manifest_uri,
        startup_script_uri: $startup_script_uri,
        quota_preflight_uri: $quota_preflight_uri
      },
      timeline: {
        launch_created_at_utc: $launch_created_at_utc,
        launch_failure_at_utc: $created_at_utc
      },
      manifest_of_manifests: {
        remote_uri: $manifest_of_manifests_uri,
        sha256: $manifest_of_manifests_sha256
      },
      retained_objects: [
        {
          artifact_kind: "launch_manifest",
          evidence_role: "manifest",
          remote_uri: $launch_manifest_uri,
          sha256: $manifest_sha256,
          byte_length: ($manifest_bytes | tonumber),
          source_mode: "local_upload"
        },
        {
          artifact_kind: "startup_script_snapshot",
          evidence_role: "artifact",
          remote_uri: $startup_script_uri,
          sha256: $startup_script_sha256,
          byte_length: ($startup_script_bytes | tonumber),
          source_mode: "local_upload"
        },
        {
          artifact_kind: "quota_preflight",
          evidence_role: "manifest",
          remote_uri: $quota_preflight_uri,
          sha256: $preflight_sha256,
          byte_length: ($preflight_bytes | tonumber),
          source_mode: "local_upload"
        },
        {
          artifact_kind: "launch_failure_log",
          evidence_role: "log",
          remote_uri: $launch_failure_log_uri,
          sha256: $launch_failure_log_sha256,
          byte_length: ($launch_failure_log_bytes | tonumber),
          source_mode: "local_upload"
        },
        {
          artifact_kind: "manifest_of_manifests",
          evidence_role: "manifest",
          remote_uri: $manifest_of_manifests_uri,
          sha256: $manifest_of_manifests_sha256,
          byte_length: null,
          source_mode: "local_upload"
        }
      ],
      detail: "The run failed before the VM became available, but the launch folder still preserves the selected profile, quota preflight, startup snapshot, and typed launch failure classification."
    }' > "${final_manifest_file}"
  gcloud storage cp --quiet "${final_manifest_file}" "${final_manifest_uri}" >/dev/null
  wait_for_object "${final_manifest_uri}"
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
lane_label="$(jq -r '.lane_label // "psion"' "${LAUNCH_FILE}")"
image_project="$(jq -r '.image_policy.image_project' "${LAUNCH_FILE}")"
image_family="$(jq -r '.image_policy.image_family' "${LAUNCH_FILE}")"
driver_posture="$(jq -r '.image_policy.driver_posture' "${LAUNCH_FILE}")"
workspace_root="$(jq -r '.startup_policy.workspace_root' "${LAUNCH_FILE}")"
pre_training_command="$(jq -r '.startup_policy.pre_training_command // empty' "${LAUNCH_FILE}")"
training_command="$(jq -r '.startup_policy.training_command' "${LAUNCH_FILE}")"
post_training_archive_command="$(jq -r '.startup_policy.post_training_archive_command // empty' "${LAUNCH_FILE}")"
post_training_restore_command="$(jq -r '.startup_policy.post_training_restore_command // empty' "${LAUNCH_FILE}")"
rust_toolchain="$(jq -r '.startup_policy.rustup_default_toolchain' "${LAUNCH_FILE}")"
apt_packages="$(jq -r '.startup_policy.package_install | join(" ")' "${LAUNCH_FILE}")"
final_manifest_object="$(jq -r '.teardown_policy.final_manifest_object' "${LAUNCH_FILE}")"
input_package_descriptor_uri="$(jq -r '.default_input_package_descriptor_uri' "${LAUNCH_FILE}")"
provisioning_model="$(jq -r '.provisioning_model' "${LAUNCH_FILE}")"
maintenance_policy="$(jq -r '.maintenance_policy' "${LAUNCH_FILE}")"
restart_on_failure="$(jq -r '.restart_on_failure' "${LAUNCH_FILE}")"
external_ip="$(jq -r '.external_ip' "${LAUNCH_FILE}")"
os_login="$(jq -r '.os_login' "${LAUNCH_FILE}")"

profile_label="$(jq -r '.profile_label' <<<"${profile_json}")"
trainer_lane_id="$(jq -r '.trainer_lane_id // "psion_reference_pilot_bundle"' <<<"${profile_json}")"
expected_execution_backend="$(jq -r '.expected_execution_backend // "cpu"' <<<"${profile_json}")"
machine_type="$(jq -r '.machine_type' <<<"${profile_json}")"
accelerator_type="$(jq -r '.accelerator_type' <<<"${profile_json}")"
accelerator_count="$(jq -r '.accelerator_count' <<<"${profile_json}")"
boot_disk_type="$(jq -r '.boot_disk_type' <<<"${profile_json}")"
boot_disk_gb="$(jq -r '.boot_disk_gb' <<<"${profile_json}")"
low_disk_watermark_gb="$(jq -r '.low_disk_watermark_gb' <<<"${profile_json}")"
declared_run_cost_ceiling_usd="$(jq -r '.declared_run_cost_ceiling_usd' <<<"${profile_json}")"
profile_input_package_descriptor_uri="$(
  jq -r '.input_package_descriptor_uri // empty' <<<"${profile_json}"
)"
profile_pre_training_command="$(jq -r '.startup_policy_overrides.pre_training_command // empty' <<<"${profile_json}")"
profile_training_command="$(jq -r '.startup_policy_overrides.training_command // empty' <<<"${profile_json}")"
profile_post_training_archive_command="$(jq -r '.startup_policy_overrides.post_training_archive_command // empty' <<<"${profile_json}")"
profile_post_training_restore_command="$(jq -r '.startup_policy_overrides.post_training_restore_command // empty' <<<"${profile_json}")"
profile_provisioning_model="$(jq -r '.provisioning_model // empty' <<<"${profile_json}")"
profile_maintenance_policy="$(jq -r '.maintenance_policy // empty' <<<"${profile_json}")"
profile_restart_on_failure="$(jq -r '.restart_on_failure // empty' <<<"${profile_json}")"
profile_external_ip="$(jq -r '.external_ip // empty' <<<"${profile_json}")"
profile_os_login="$(jq -r '.os_login // empty' <<<"${profile_json}")"
if [[ -n "${profile_pre_training_command}" ]]; then
  pre_training_command="${profile_pre_training_command}"
fi
if [[ -n "${profile_input_package_descriptor_uri}" ]]; then
  input_package_descriptor_uri="${profile_input_package_descriptor_uri}"
fi
if [[ -n "${profile_training_command}" ]]; then
  training_command="${profile_training_command}"
fi
if [[ -n "${profile_post_training_archive_command}" ]]; then
  post_training_archive_command="${profile_post_training_archive_command}"
fi
if [[ "${profile_post_training_restore_command}" == "__none__" ]]; then
  post_training_restore_command=""
elif [[ -n "${profile_post_training_restore_command}" ]]; then
  post_training_restore_command="${profile_post_training_restore_command}"
fi
if [[ -n "${profile_provisioning_model}" ]]; then
  provisioning_model="${profile_provisioning_model}"
fi
if [[ -n "${profile_maintenance_policy}" ]]; then
  maintenance_policy="${profile_maintenance_policy}"
fi
if [[ -n "${profile_restart_on_failure}" ]]; then
  restart_on_failure="${profile_restart_on_failure}"
fi
if [[ -n "${profile_external_ip}" ]]; then
  external_ip="${profile_external_ip}"
fi
if [[ -n "${profile_os_login}" ]]; then
  os_login="${profile_os_login}"
fi

timestamp_tag="$(date -u '+%Y%m%dt%H%M%Sz' | tr '[:upper:]' '[:lower:]')"
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="psion-${profile_label}-${timestamp_tag}"
fi
if [[ -z "${INSTANCE_NAME}" ]]; then
  INSTANCE_NAME="${RUN_ID}"
fi
if [[ -n "${INPUT_PACKAGE_DESCRIPTOR_URI_OVERRIDE}" ]]; then
  input_package_descriptor_uri="${INPUT_PACKAGE_DESCRIPTOR_URI_OVERRIDE}"
fi
if [[ "${PRE_TRAINING_COMMAND_OVERRIDE}" == "__none__" ]]; then
  pre_training_command=""
elif [[ -n "${PRE_TRAINING_COMMAND_OVERRIDE}" ]]; then
  pre_training_command="${PRE_TRAINING_COMMAND_OVERRIDE}"
fi
if [[ -n "${TRAINING_COMMAND_OVERRIDE}" ]]; then
  training_command="${TRAINING_COMMAND_OVERRIDE}"
fi
if [[ "${POST_TRAINING_ARCHIVE_COMMAND_OVERRIDE}" == "__none__" ]]; then
  post_training_archive_command=""
elif [[ -n "${POST_TRAINING_ARCHIVE_COMMAND_OVERRIDE}" ]]; then
  post_training_archive_command="${POST_TRAINING_ARCHIVE_COMMAND_OVERRIDE}"
fi
if [[ "${POST_TRAINING_RESTORE_COMMAND_OVERRIDE}" == "__none__" ]]; then
  post_training_restore_command=""
elif [[ -n "${POST_TRAINING_RESTORE_COMMAND_OVERRIDE}" ]]; then
  post_training_restore_command="${POST_TRAINING_RESTORE_COMMAND_OVERRIDE}"
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
wait_for_object "${input_package_descriptor_uri}"
input_package_descriptor_json="$(gcloud storage cat "${input_package_descriptor_uri}")"
input_package_archive_uri="$(jq -r '.archive_uri' <<<"${input_package_descriptor_json}")"
input_package_archive_sha256="$(jq -r '.archive_sha256' <<<"${input_package_descriptor_json}")"
input_package_manifest_sha256="$(jq -r '.manifest_sha256' <<<"${input_package_descriptor_json}")"
input_materialization_mode="$(jq -r '.materialization.mode' <<<"${input_package_descriptor_json}")"
input_materialization_max_local_gb="$(jq -r '.materialization.max_local_working_set_gb' <<<"${input_package_descriptor_json}")"
wait_for_object "${input_package_archive_uri}"

manifest_file="${tmpdir}/psion_google_single_node_launch_manifest.json"
pre_training_command_file="${tmpdir}/pre_training_command.sh"
training_command_file="${tmpdir}/training_command.sh"
apt_packages_file="${tmpdir}/apt_packages.txt"
post_training_archive_command_file="${tmpdir}/post_training_archive_command.sh"
post_training_restore_command_file="${tmpdir}/post_training_restore_command.sh"
printf '%s\n' "${pre_training_command}" > "${pre_training_command_file}"
printf '%s\n' "${training_command}" > "${training_command_file}"
printf '%s\n' "${apt_packages}" > "${apt_packages_file}"
printf '%s\n' "${post_training_archive_command}" > "${post_training_archive_command_file}"
printf '%s\n' "${post_training_restore_command}" > "${post_training_restore_command_file}"

jq -n \
  --arg schema_version "psion.google_single_node_launch_manifest.v1" \
  --arg created_at_utc "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  --arg run_id "${RUN_ID}" \
  --arg instance_name "${INSTANCE_NAME}" \
  --arg project_id "${PROJECT_ID}" \
  --arg profile_id "${PROFILE_ID}" \
  --arg profile_label "${profile_label}" \
  --arg trainer_lane_id "${trainer_lane_id}" \
  --arg expected_execution_backend "${expected_execution_backend}" \
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
  --arg input_package_descriptor_uri "${input_package_descriptor_uri}" \
  --arg input_package_archive_uri "${input_package_archive_uri}" \
  --arg input_package_archive_sha256 "${input_package_archive_sha256}" \
  --arg input_package_manifest_sha256 "${input_package_manifest_sha256}" \
  --arg input_materialization_mode "${input_materialization_mode}" \
  --argjson input_materialization_max_local_gb "${input_materialization_max_local_gb}" \
  --argjson input_benchmark_execution_posture "$(jq '.benchmark_execution_posture' <<<"${input_package_descriptor_json}")" \
  --argjson input_key_digests "$(jq '.key_digests' <<<"${input_package_descriptor_json}")" \
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
  --arg pre_training_command "${pre_training_command}" \
  --arg training_command "${training_command}" \
  --arg post_training_archive_command "${post_training_archive_command}" \
  --arg post_training_restore_command "${post_training_restore_command}" \
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
  --arg provisioning_model "${provisioning_model}" \
  --arg maintenance_policy "${maintenance_policy}" \
  --argjson restart_on_failure "${restart_on_failure}" \
  --argjson external_ip "${external_ip}" \
  --argjson os_login "${os_login}" \
  --argjson quota_preflight "$(cat "${preflight_report_file}")" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    instance_name: $instance_name,
    project_id: $project_id,
    profile_id: $profile_id,
    profile_label: $profile_label,
    trainer_lane_id: $trainer_lane_id,
    expected_execution_backend: $expected_execution_backend,
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
    input_package: {
      descriptor_uri: $input_package_descriptor_uri,
      archive_uri: $input_package_archive_uri,
      archive_sha256: $input_package_archive_sha256,
      manifest_sha256: $input_package_manifest_sha256,
      materialization_mode: $input_materialization_mode,
      max_local_working_set_gb: $input_materialization_max_local_gb,
      benchmark_execution_posture: $input_benchmark_execution_posture,
      key_digests: $input_key_digests
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
      trainer_lane_id: $trainer_lane_id,
      expected_execution_backend: $expected_execution_backend,
      pre_training_command: (if $pre_training_command == "" then null else $pre_training_command end),
      command: $training_command,
      post_training_archive_command: (if $post_training_archive_command == "" then null else $post_training_archive_command end),
      post_training_restore_command: (if $post_training_restore_command == "" then null else $post_training_restore_command end),
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

restart_on_failure_flag="--no-restart-on-failure"
if [[ "${restart_on_failure}" == "true" ]]; then
  restart_on_failure_flag="--restart-on-failure"
fi
network_interface_arg="network=${network},subnet=${subnetwork},no-address"
if [[ "${external_ip}" == "true" ]]; then
  network_interface_arg="network=${network},subnet=${subnetwork}"
fi
os_login_metadata="FALSE"
if [[ "${os_login}" == "true" ]]; then
  os_login_metadata="TRUE"
fi

create_stdout_file="${tmpdir}/gcloud-create.stdout"
create_stderr_file="${tmpdir}/gcloud-create.stderr"
metadata_from_file_arg="startup-script=${STARTUP_SCRIPT},psion-training-command=${training_command_file},psion-apt-packages=${apt_packages_file}"
if [[ -n "${pre_training_command}" ]]; then
  metadata_from_file_arg="${metadata_from_file_arg},psion-pre-training-command=${pre_training_command_file}"
fi
if [[ -n "${post_training_archive_command}" ]]; then
  metadata_from_file_arg="${metadata_from_file_arg},psion-post-training-archive-command=${post_training_archive_command_file}"
fi
if [[ -n "${post_training_restore_command}" ]]; then
  metadata_from_file_arg="${metadata_from_file_arg},psion-post-training-restore-command=${post_training_restore_command_file}"
fi
if ! gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${selected_zone}" \
  --machine-type="${machine_type}" \
  --accelerator="type=${accelerator_type},count=${accelerator_count}" \
  --maintenance-policy="${maintenance_policy}" \
  --provisioning-model="${provisioning_model}" \
  "${restart_on_failure_flag}" \
  --network-interface="${network_interface_arg}" \
  --service-account="${service_account_email}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --tags="${target_tags}" \
  --labels="lane=$(sanitize_label "${lane_label}"),purpose=train,track=single-node,profile=$(sanitize_label "${profile_label}")" \
  --boot-disk-size="${boot_disk_gb}GB" \
  --boot-disk-type="${boot_disk_type}" \
  --image="${image_name}" \
  --image-project="${image_project}" \
  --metadata="enable-oslogin=${os_login_metadata},psion-run-id=${RUN_ID},psion-bucket-url=${bucket_url},psion-repo-clone-url=${repo_clone_url},psion-git-revision=${git_revision},psion-workspace-root=${workspace_root},psion-output-subdir=output,psion-log-subdir=logs,psion-scratch-subdir=scratch,psion-repo-subdir=repo,psion-low-disk-watermark-gb=${low_disk_watermark_gb},psion-rust-toolchain=${rust_toolchain},psion-input-package-descriptor-uri=${input_package_descriptor_uri},psion-input-package-archive-uri=${input_package_archive_uri},psion-input-package-archive-sha256=${input_package_archive_sha256},psion-input-package-manifest-sha256=${input_package_manifest_sha256},psion-input-materialization-mode=${input_materialization_mode},psion-launch-manifest-uri=${manifest_uri}" \
  --metadata-from-file="${metadata_from_file_arg}" >"${create_stdout_file}" 2>"${create_stderr_file}"; then
  failure_detail="$(tr '\n' ' ' < "${create_stderr_file}" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
  write_launch_failure_manifest "launch_capacity_failure" "${failure_detail}" "${create_stderr_file}"
  cat "${create_stderr_file}" >&2
  exit 1
fi

echo "instance launched:"
gcloud compute instances describe "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${selected_zone}" \
  --format=json | jq '{name, zone: (.zone | split("/") | last), machineType: (.machineType | split("/") | last), guestAccelerators, serviceAccounts, labels, status}'

echo "launch manifest uri: ${manifest_uri}"
echo "iap ssh: gcloud compute ssh ${INSTANCE_NAME} --project=${PROJECT_ID} --zone=${selected_zone} --tunnel-through-iap"

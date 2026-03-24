#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

launch_file="${repo_root}/fixtures/parameter_golf/runpod/parameter_golf_runpod_8xh100_launch_profiles_v1.json"
preflight_script="${repo_root}/scripts/parameter-golf-runpod-operator-preflight.sh"

profile_id=""
run_id=""
pod_host=""
pod_port=""
ssh_user="root"
ssh_key_path=""
manifest_only=false

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-launch-8xh100.sh [options]

Options:
  --profile <profile_id>   Launch profile from the committed RunPod authority file.
  --run-id <run_id>        Stable run identifier for the manifest.
  --pod-host <host>        RunPod pod host or IP for later real execution.
  --pod-port <port>        SSH port for later real execution.
  --ssh-user <user>        SSH user for later real execution. Default: root.
  --ssh-key <path>         SSH private key path for later real execution.
  --manifest-only          Emit the launch manifest without contacting a pod.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile_id="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --pod-host)
      pod_host="$2"
      shift 2
      ;;
    --pod-port)
      pod_port="$2"
      shift 2
      ;;
    --ssh-user)
      ssh_user="$2"
      shift 2
      ;;
    --ssh-key)
      ssh_key_path="$2"
      shift 2
      ;;
    --manifest-only)
      manifest_only=true
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

if [[ -z "${profile_id}" ]]; then
  echo "error: --profile is required" >&2
  usage
  exit 1
fi

if [[ -z "${run_id}" ]]; then
  run_id="parameter-golf-runpod-8xh100-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ "${manifest_only}" != "true" && ( -z "${pod_host}" || -z "${pod_port}" ) ]]; then
  echo "error: real RunPod launch requires --pod-host and --pod-port" >&2
  exit 1
fi

preflight_json="$("${preflight_script}" --profile "${profile_id}")"
profile_json="$(jq -c --arg profile_id "${profile_id}" '.profiles[] | select(.profile_id == $profile_id)' "${launch_file}")"
if [[ -z "${profile_json}" ]]; then
  echo "error: launch profile ${profile_id} was not found" >&2
  exit 1
fi

workspace_root="$(jq -r '.workspace_root' "${launch_file}")"
default_branch="$(jq -r '.default_branch' "${launch_file}")"
descriptor_uri="$(jq -r '.default_input_package_descriptor_uri' "${launch_file}")"
submission_dir="${workspace_root}/parameter-golf-runpod/${run_id}/exported_submission/records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2"

jq -n \
  --arg schema_version "parameter_golf.runpod_8xh100_launch_manifest.v1" \
  --arg created_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg run_id "${run_id}" \
  --arg profile_id "${profile_id}" \
  --arg profile_label "$(jq -r '.profile_label' <<<"${profile_json}")" \
  --arg trainer_lane_id "$(jq -r '.trainer_lane_id' <<<"${profile_json}")" \
  --arg expected_execution_backend "$(jq -r '.expected_execution_backend' <<<"${profile_json}")" \
  --arg branch "${default_branch}" \
  --arg repo_clone_url "$(jq -r '.repo_clone_url' "${launch_file}")" \
  --arg workspace_root "${workspace_root}" \
  --arg pod_host "${pod_host}" \
  --arg pod_port "${pod_port}" \
  --arg ssh_user "${ssh_user}" \
  --arg ssh_key_path "${ssh_key_path}" \
  --arg descriptor_uri "${descriptor_uri}" \
  --arg descriptor_artifact_ref "$(jq -r '.default_input_package_descriptor_artifact_ref' "${launch_file}")" \
  --arg submission_dir "${submission_dir}" \
  --arg pod_shape "$(jq -r '.pod_shape' <<<"${profile_json}")" \
  --arg runtime_image_posture "$(jq -r '.runtime_image_posture' <<<"${profile_json}")" \
  --arg workspace_mount_posture "$(jq -r '.workspace_mount_posture' <<<"${profile_json}")" \
  --arg accelerator_type "$(jq -r '.accelerator_type' <<<"${profile_json}")" \
  --argjson accelerator_count "$(jq -r '.accelerator_count' <<<"${profile_json}")" \
  --argjson require_non_mig "$(jq -r '.require_non_mig' <<<"${profile_json}")" \
  --argjson world_size "$(jq -r '.world_size' <<<"${profile_json}")" \
  --argjson grad_accum_steps "$(jq -r '.grad_accum_steps' <<<"${profile_json}")" \
  --arg pre_training_command "$(jq -r '.pre_training_command' <<<"${profile_json}")" \
  --arg execution_entrypoint_command "$(jq -r '.execution_entrypoint_command' <<<"${profile_json}")" \
  --arg finalizer_command "$(jq -r '.finalizer_command' <<<"${profile_json}")" \
  --argjson expected_receipt_paths "$(jq -c '.expected_receipt_paths' <<<"${profile_json}")" \
  --argjson preflight "${preflight_json}" \
  --arg manifest_only "${manifest_only}" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    provider: "runpod",
    run_id: $run_id,
    profile_id: $profile_id,
    profile_label: $profile_label,
    trainer_lane_id: $trainer_lane_id,
    expected_execution_backend: $expected_execution_backend,
    git_ref: {
      repo_clone_url: $repo_clone_url,
      branch: $branch
    },
    operator_endpoint: {
      ssh_user: $ssh_user,
      pod_host: (if $pod_host == "" then null else $pod_host end),
      pod_port: (if $pod_port == "" then null else ($pod_port | tonumber) end),
      ssh_key_path: (if $ssh_key_path == "" then null else $ssh_key_path end)
    },
    machine: {
      pod_shape: $pod_shape,
      runtime_image_posture: $runtime_image_posture,
      workspace_mount_posture: $workspace_mount_posture
    },
    topology: {
      accelerator_type: $accelerator_type,
      accelerator_count: $accelerator_count,
      require_non_mig: $require_non_mig,
      world_size: $world_size,
      grad_accum_steps: $grad_accum_steps
    },
    input_package: {
      descriptor_uri: $descriptor_uri,
      descriptor_artifact_ref: $descriptor_artifact_ref
    },
    run_roots: {
      workspace_root: $workspace_root,
      run_root: ($workspace_root + "/parameter-golf-runpod/" + $run_id),
      submission_dir: $submission_dir
    },
    commands: {
      pre_training_command: $pre_training_command,
      execution_entrypoint_command: $execution_entrypoint_command,
      finalizer_command: $finalizer_command
    },
    expected_receipt_paths: $expected_receipt_paths,
    preflight: $preflight,
    manifest_only: ($manifest_only == "true")
  }'

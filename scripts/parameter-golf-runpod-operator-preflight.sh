#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

profile_id=""

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-runpod-operator-preflight.sh --profile <profile_id>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile_id="$2"
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

if [[ -z "${profile_id}" ]]; then
  echo "error: --profile is required" >&2
  usage
  exit 1
fi

policy_file="${repo_root}/fixtures/parameter_golf/runpod/parameter_golf_runpod_8xh100_operator_preflight_policy_v1.json"
launch_file="${repo_root}/fixtures/parameter_golf/runpod/parameter_golf_runpod_8xh100_launch_profiles_v1.json"
guardrail_file="${repo_root}/fixtures/parameter_golf/runpod/parameter_golf_runpod_8xh100_cost_guardrails_v1.json"

for command in $(jq -r '.required_commands[]' "${policy_file}"); do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "error: required command ${command} is missing" >&2
    exit 1
  fi
done

for artifact in $(jq -r '.required_local_artifacts[]' "${policy_file}"); do
  if [[ ! -e "${repo_root}/${artifact}" ]]; then
    echo "error: required local artifact ${artifact} is missing" >&2
    exit 1
  fi
done

if ! jq -e --arg profile_id "${profile_id}" '.supported_profiles | index($profile_id)' "${policy_file}" >/dev/null; then
  echo "error: unsupported profile ${profile_id}" >&2
  exit 1
fi

profile_json="$(jq -c --arg profile_id "${profile_id}" '.profiles[] | select(.profile_id == $profile_id)' "${launch_file}")"
if [[ -z "${profile_json}" ]]; then
  echo "error: launch profile ${profile_id} was not found" >&2
  exit 1
fi

accelerator_count="$(jq -r '.accelerator_count' <<<"${profile_json}")"
world_size="$(jq -r '.world_size' <<<"${profile_json}")"
grad_accum_steps="$(jq -r '.grad_accum_steps' <<<"${profile_json}")"
declared_run_cost_ceiling_usd="$(jq -r '.declared_run_cost_ceiling_usd' <<<"${profile_json}")"
max_runtime_hours="$(jq -r '.max_runtime_hours' <<<"${profile_json}")"
guardrail_cost_ceiling="$(jq -r '.declared_run_cost_ceiling_usd' "${guardrail_file}")"
guardrail_runtime_ceiling="$(jq -r '.max_runtime_hours' "${guardrail_file}")"

if [[ "${accelerator_count}" != "8" ]]; then
  echo "error: RunPod PGOLF lane requires accelerator_count=8, found ${accelerator_count}" >&2
  exit 1
fi

if [[ "${world_size}" != "8" ]]; then
  echo "error: RunPod PGOLF lane requires world_size=8, found ${world_size}" >&2
  exit 1
fi

if [[ "${grad_accum_steps}" != "1" ]]; then
  echo "error: RunPod PGOLF lane requires grad_accum_steps=1, found ${grad_accum_steps}" >&2
  exit 1
fi

if [[ "${declared_run_cost_ceiling_usd}" != "${guardrail_cost_ceiling}" ]]; then
  echo "error: launch profile cost ceiling ${declared_run_cost_ceiling_usd} drifted from guardrail ${guardrail_cost_ceiling}" >&2
  exit 1
fi

if [[ "${max_runtime_hours}" != "${guardrail_runtime_ceiling}" ]]; then
  echo "error: launch profile runtime ceiling ${max_runtime_hours} drifted from guardrail ${guardrail_runtime_ceiling}" >&2
  exit 1
fi

jq -n \
  --arg schema_version "parameter_golf.runpod_8xh100_operator_preflight.v1" \
  --arg policy_id "$(jq -r '.policy_id' "${policy_file}")" \
  --arg profile_id "${profile_id}" \
  --arg trainer_lane_id "$(jq -r '.trainer_lane_id' <<<"${profile_json}")" \
  --arg result "ready" \
  --arg accelerator_type "$(jq -r '.accelerator_type' <<<"${profile_json}")" \
  --argjson accelerator_count "${accelerator_count}" \
  --argjson world_size "${world_size}" \
  --argjson grad_accum_steps "${grad_accum_steps}" \
  --argjson declared_run_cost_ceiling_usd "${declared_run_cost_ceiling_usd}" \
  --argjson max_runtime_hours "${max_runtime_hours}" \
  --arg descriptor_uri "$(jq -r '.default_input_package_descriptor_uri' "${launch_file}")" \
  '{
    schema_version: $schema_version,
    policy_id: $policy_id,
    profile_id: $profile_id,
    trainer_lane_id: $trainer_lane_id,
    result: $result,
    accelerator_type: $accelerator_type,
    accelerator_count: $accelerator_count,
    world_size: $world_size,
    grad_accum_steps: $grad_accum_steps,
    declared_run_cost_ceiling_usd: $declared_run_cost_ceiling_usd,
    max_runtime_hours: $max_runtime_hours,
    input_package_descriptor_uri: $descriptor_uri,
    remote_inventory_validation: "deferred_to_real_pod_launch",
    detail: "The committed RunPod 8xH100 lane is locally consistent and ready for manifest-only launch or later real pod execution."
  }'

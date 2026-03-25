#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
POLICY_FILE="${POLICY_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_two_node_swarm_operator_preflight_policy_v1.json}"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

PROJECT_ID="${PROJECT_ID:-$(jq -r '.project_id' "${POLICY_FILE}")}"
PAIR_ID="${PAIR_ID:-}"

usage() {
  cat <<'EOF'
Usage: psion-google-quota-preflight-two-node-swarm.sh [options]

Options:
  --pair <pair_id>    Force one admitted zone pair instead of auto-selecting the first ready pair.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pair)
      PAIR_ID="$2"
      shift 2
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

metric_summary() {
  local region_json="$1"
  local metric="$2"
  jq -c --arg metric "${metric}" '
    first(.quotas[] | select(.metric == $metric)) as $quota
    | {
        metric: $metric,
        limit: ($quota.limit // 0),
        usage: ($quota.usage // 0),
        available: (($quota.limit // 0) - ($quota.usage // 0))
      }
  ' <<<"${region_json}"
}

pair_report() {
  local pair_json="$1"
  local region_json="$2"
  local machine_type accelerator_type coordinator_zone contributor_zone
  local cpu_metric gpu_metric instance_metric disk_metric
  local required_vcpus required_memory required_instances required_accelerators required_disk
  local accelerator_count_per_node
  local coordinator_machine_state contributor_machine_state coordinator_accelerator_state contributor_accelerator_state
  local cpu_summary gpu_summary instance_summary disk_summary
  local result failure_reasons_json

  machine_type="$(jq -r '.machine_type' <<<"${pair_json}")"
  accelerator_type="$(jq -r '.accelerator_type' <<<"${pair_json}")"
  accelerator_count_per_node="$(jq -r '.accelerator_count_per_node' <<<"${pair_json}")"
  coordinator_zone="$(jq -r '.coordinator_zone' <<<"${pair_json}")"
  contributor_zone="$(jq -r '.contributor_zone' <<<"${pair_json}")"
  cpu_metric="$(jq -r '.cpu_quota_metric' <<<"${pair_json}")"
  gpu_metric="$(jq -r '.gpu_quota_metric' <<<"${pair_json}")"
  instance_metric="$(jq -r '.instance_quota_metric' <<<"${pair_json}")"
  disk_metric="$(jq -r '.disk_quota_metric' <<<"${pair_json}")"
  required_vcpus="$(jq -r '.required_vcpus_total' <<<"${pair_json}")"
  required_memory="$(jq -r '.required_memory_mb_total' <<<"${pair_json}")"
  required_instances="$(jq -r '.required_instances_total' <<<"${pair_json}")"
  required_accelerators="$(jq -r '.required_accelerators_total' <<<"${pair_json}")"
  required_disk="$(jq -r '.required_boot_disk_gb_total' <<<"${pair_json}")"

  if gcloud compute machine-types describe "${machine_type}" --zone="${coordinator_zone}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    coordinator_machine_state="available"
  else
    coordinator_machine_state="missing"
  fi
  if gcloud compute machine-types describe "${machine_type}" --zone="${contributor_zone}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    contributor_machine_state="available"
  else
    contributor_machine_state="missing"
  fi
  if gcloud compute accelerator-types describe "${accelerator_type}" --zone="${coordinator_zone}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    coordinator_accelerator_state="available"
  else
    coordinator_accelerator_state="missing"
  fi
  if gcloud compute accelerator-types describe "${accelerator_type}" --zone="${contributor_zone}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    contributor_accelerator_state="available"
  else
    contributor_accelerator_state="missing"
  fi

  cpu_summary="$(metric_summary "${region_json}" "${cpu_metric}")"
  gpu_summary="$(metric_summary "${region_json}" "${gpu_metric}")"
  instance_summary="$(metric_summary "${region_json}" "${instance_metric}")"
  disk_summary="$(metric_summary "${region_json}" "${disk_metric}")"

  failure_reasons_json="$(
    jq -nc \
      --arg coordinator_machine_state "${coordinator_machine_state}" \
      --arg contributor_machine_state "${contributor_machine_state}" \
      --arg coordinator_accelerator_state "${coordinator_accelerator_state}" \
      --arg contributor_accelerator_state "${contributor_accelerator_state}" \
      --argjson cpu_summary "${cpu_summary}" \
      --argjson gpu_summary "${gpu_summary}" \
      --argjson instance_summary "${instance_summary}" \
      --argjson disk_summary "${disk_summary}" \
      --argjson required_vcpus "${required_vcpus}" \
      --argjson required_accelerators "${required_accelerators}" \
      --argjson required_instances "${required_instances}" \
      --argjson required_disk "${required_disk}" '
      [
        if $coordinator_machine_state != "available" then "coordinator_machine_type_unavailable" else empty end,
        if $contributor_machine_state != "available" then "contributor_machine_type_unavailable" else empty end,
        if $coordinator_accelerator_state != "available" then "coordinator_accelerator_unavailable" else empty end,
        if $contributor_accelerator_state != "available" then "contributor_accelerator_unavailable" else empty end,
        if $cpu_summary.available < $required_vcpus then "cpu_quota_insufficient" else empty end,
        if $gpu_summary.available < $required_accelerators then "gpu_quota_insufficient" else empty end,
        if $instance_summary.available < $required_instances then "instance_quota_insufficient" else empty end,
        if $disk_summary.available < $required_disk then "disk_quota_insufficient" else empty end
      ]'
  )"

  if [[ "${failure_reasons_json}" == "[]" ]]; then
    result="ready"
  else
    result="blocked"
  fi

  jq -nc \
    --arg pair_id "$(jq -r '.pair_id' <<<"${pair_json}")" \
    --arg coordinator_zone "${coordinator_zone}" \
    --arg contributor_zone "${contributor_zone}" \
    --arg coordinator_profile_id "$(jq -r '.coordinator_profile_id' <<<"${pair_json}")" \
    --arg contributor_profile_id "$(jq -r '.contributor_profile_id' <<<"${pair_json}")" \
    --arg machine_type "${machine_type}" \
    --arg accelerator_type "${accelerator_type}" \
    --arg result "${result}" \
    --argjson accelerator_count_per_node "${accelerator_count_per_node}" \
    --argjson required_vcpus_total "${required_vcpus}" \
    --argjson required_memory_mb_total "${required_memory}" \
    --argjson required_instances_total "${required_instances}" \
    --argjson required_accelerators_total "${required_accelerators}" \
    --argjson required_boot_disk_gb_total "${required_disk}" \
    --argjson cpu_summary "${cpu_summary}" \
    --argjson gpu_summary "${gpu_summary}" \
    --argjson instance_summary "${instance_summary}" \
    --argjson disk_summary "${disk_summary}" \
    --arg coordinator_machine_state "${coordinator_machine_state}" \
    --arg contributor_machine_state "${contributor_machine_state}" \
    --arg coordinator_accelerator_state "${coordinator_accelerator_state}" \
    --arg contributor_accelerator_state "${contributor_accelerator_state}" \
    --argjson failure_reasons "${failure_reasons_json}" \
    '{
      pair_id: $pair_id,
      coordinator_zone: $coordinator_zone,
      contributor_zone: $contributor_zone,
      coordinator_profile_id: $coordinator_profile_id,
      contributor_profile_id: $contributor_profile_id,
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count_per_node: $accelerator_count_per_node,
      required_vcpus_total: $required_vcpus_total,
      required_memory_mb_total: $required_memory_mb_total,
      required_instances_total: $required_instances_total,
      required_accelerators_total: $required_accelerators_total,
      required_boot_disk_gb_total: $required_boot_disk_gb_total,
      node_availability: {
        coordinator_machine_type_state: $coordinator_machine_state,
        contributor_machine_type_state: $contributor_machine_state,
        coordinator_accelerator_state: $coordinator_accelerator_state,
        contributor_accelerator_state: $contributor_accelerator_state
      },
      quotas: {
        cpu: $cpu_summary,
        accelerator: $gpu_summary,
        instances: $instance_summary,
        disk: $disk_summary
      },
      result: $result,
      failure_reasons: $failure_reasons
    }'
}

pair_candidates_json="$(jq -c '.supported_pairs' "${POLICY_FILE}")"
if [[ -n "${PAIR_ID}" ]]; then
  pair_candidates_json="$(jq -c --arg pair_id "${PAIR_ID}" '[.supported_pairs[] | select(.pair_id == $pair_id)]' "${POLICY_FILE}")"
fi

if [[ "${pair_candidates_json}" == "[]" ]]; then
  echo "error: no supported zone pair matched `${PAIR_ID}`" >&2
  exit 1
fi

region_json="$(
  gcloud compute regions describe "$(jq -r '.supported_pairs[0].coordinator_zone' "${POLICY_FILE}" | sed 's/-[a-z]$//')" \
    --project="${PROJECT_ID}" \
    --format=json
)"

pair_reports_json='[]'
while IFS= read -r pair_json; do
  pair_report_json="$(pair_report "${pair_json}" "${region_json}")"
  pair_reports_json="$(jq -c --argjson report "${pair_report_json}" '. + [$report]' <<<"${pair_reports_json}")"
done < <(jq -c '.[]' <<<"${pair_candidates_json}")

selected_pair_json="$(jq -c '([.[] | select(.result == "ready")] | first) // first' <<<"${pair_reports_json}")"
overall_result="$(jq -r 'if any(.[]; .result == "ready") then "ready" else "blocked" end' <<<"${pair_reports_json}")"

jq -n \
  --arg schema_version "psion.google_two_node_swarm_quota_preflight.v1" \
  --arg checked_at_utc "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  --arg project_id "${PROJECT_ID}" \
  --arg pair_override "${PAIR_ID}" \
  --arg result "${overall_result}" \
  --argjson pair_results "${pair_reports_json}" \
  --argjson selected_pair "${selected_pair_json}" \
  '{
    schema_version: $schema_version,
    checked_at_utc: $checked_at_utc,
    project_id: $project_id,
    pair_override: (if $pair_override == "" then null else $pair_override end),
    pair_results: $pair_results,
    selected_pair: $selected_pair,
    result: $result
  }'

if [[ "${overall_result}" != "ready" ]]; then
  exit 1
fi

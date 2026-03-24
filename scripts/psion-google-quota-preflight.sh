#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
PROFILE_ID="${PROFILE_ID:-g2_l4_single_node_accelerated}"
ZONE="${ZONE:-}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
GUARDRAIL_FILE="${GUARDRAIL_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_billing_guardrails_v1.json}"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

profile_json="$(
  jq -c --arg profile_id "${PROFILE_ID}" '.quota_preflight.profiles[] | select(.profile_id == $profile_id)' "${GUARDRAIL_FILE}"
)"

if [[ -z "${profile_json}" ]]; then
  echo "error: unknown profile ${PROFILE_ID}" >&2
  exit 1
fi

region="$(jq -r '.quota_preflight.region' "${GUARDRAIL_FILE}")"
instance_quota_metric="$(jq -r '.quota_preflight.instance_quota_metric' "${GUARDRAIL_FILE}")"
disk_quota_metric="$(jq -r '.quota_preflight.disk_quota_metric' "${GUARDRAIL_FILE}")"

if [[ -z "${ZONE}" ]]; then
  ZONE="$(jq -r '.zone_fallback_order[0]' <<<"${profile_json}")"
fi

machine_type="$(jq -r '.machine_type' <<<"${profile_json}")"
accelerator_type="$(jq -r '.accelerator_type' <<<"${profile_json}")"
accelerator_count="$(jq -r '.accelerator_count' <<<"${profile_json}")"
required_vcpus="$(jq -r '.required_vcpus // 0' <<<"${profile_json}")"
required_memory_mb="$(jq -r '.required_memory_mb' <<<"${profile_json}")"
boot_disk_gb="$(jq -r '.boot_disk_gb' <<<"${profile_json}")"
declared_run_cost_ceiling_usd="$(jq -r '.declared_run_cost_ceiling_usd' <<<"${profile_json}")"
max_runtime_hours="$(jq -r '.max_runtime_hours' <<<"${profile_json}")"
max_launch_attempts="$(jq -r '.max_launch_attempts' <<<"${profile_json}")"
capacity_failure_result="$(jq -r '.capacity_failure_result' <<<"${profile_json}")"
budget_amount_usd="$(jq -r '.budget.amount_usd' "${GUARDRAIL_FILE}")"
topic_name="$(jq -r '.budget.pubsub_topic' "${GUARDRAIL_FILE}")"

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

metric_candidates_json() {
  local region_json="$1"
  local metrics_json="$2"
  jq -nc \
    --argjson region "${region_json}" \
    --argjson metrics "${metrics_json}" \
    '[
      $metrics[]
      | . as $metric
      | (
          first($region.quotas[] | select(.metric == $metric))
          // {metric: $metric, limit: 0, usage: 0}
        ) as $quota
      | {
          metric: $metric,
          limit: ($quota.limit // 0),
          usage: ($quota.usage // 0),
          available: (($quota.limit // 0) - ($quota.usage // 0))
        }
    ]'
}

selected_metric_summary() {
  local candidates_json="$1"
  local required="$2"
  jq -c --argjson required "${required}" \
    '([.[] | select(.available >= $required)] | first) // (first // null)' \
    <<<"${candidates_json}"
}

metric_ready() {
  local summary_json="$1"
  local required="$2"
  if [[ "${summary_json}" == "null" ]]; then
    printf '%s' "false"
    return 0
  fi
  jq -r --argjson required "${required}" '.available >= $required' <<<"${summary_json}"
}

metric_list_json() {
  local profile_json="$1"
  local plural_key="$2"
  local singular_key="$3"
  jq -c \
    --arg plural_key "${plural_key}" \
    --arg singular_key "${singular_key}" \
    '
    if has($plural_key) then
      .[$plural_key]
    elif has($singular_key) and .[$singular_key] != null and .[$singular_key] != "" then
      [.[ $singular_key ]]
    else
      []
    end
    ' <<<"${profile_json}"
}

is_available() {
  local summary_json="$1"
  local required="$2"
  jq -r --argjson required "${required}" '.available >= $required' <<<"${summary_json}"
}

machine_type_state="available"
accelerator_state="available"

if ! machine_json="$(gcloud compute machine-types describe "${machine_type}" --zone="${ZONE}" --project="${PROJECT_ID}" --format=json 2>/dev/null)"; then
  machine_type_state="missing"
  machine_json='{}'
fi

if ! accelerator_json="$(gcloud compute accelerator-types describe "${accelerator_type}" --zone="${ZONE}" --project="${PROJECT_ID}" --format=json 2>/dev/null)"; then
  accelerator_state="missing"
  accelerator_json='{}'
fi

region_json="$(gcloud compute regions describe "${region}" --project="${PROJECT_ID}" --format=json)"
cpu_quota_metrics_json="$(metric_list_json "${profile_json}" "cpu_quota_metrics" "cpu_quota_metric")"
gpu_quota_metrics_json="$(metric_list_json "${profile_json}" "gpu_quota_metrics" "gpu_quota_metric")"
if [[ "${cpu_quota_metrics_json}" == "[]" || "${required_vcpus}" == "0" ]]; then
  cpu_candidates_json='[]'
  cpu_summary='null'
  cpu_ready="true"
else
  cpu_candidates_json="$(metric_candidates_json "${region_json}" "${cpu_quota_metrics_json}")"
  cpu_summary="$(selected_metric_summary "${cpu_candidates_json}" "${required_vcpus}")"
  cpu_ready="$(metric_ready "${cpu_summary}" "${required_vcpus}")"
fi
gpu_candidates_json="$(metric_candidates_json "${region_json}" "${gpu_quota_metrics_json}")"
gpu_summary="$(selected_metric_summary "${gpu_candidates_json}" "${accelerator_count}")"
instance_summary="$(metric_summary "${region_json}" "${instance_quota_metric}")"
disk_summary="$(metric_summary "${region_json}" "${disk_quota_metric}")"
gpu_ready="$(metric_ready "${gpu_summary}" "${accelerator_count}")"
instance_ready="$(is_available "${instance_summary}" 1)"
disk_ready="$(is_available "${disk_summary}" "${boot_disk_gb}")"

result="ready"
failure_reason=""

if [[ "${machine_type_state}" != "available" ]]; then
  result="blocked"
  failure_reason="machine_type_unavailable"
elif [[ "${accelerator_state}" != "available" ]]; then
  result="blocked"
  failure_reason="accelerator_unavailable"
elif [[ "${cpu_ready}" != "true" ]]; then
  result="blocked"
  failure_reason="cpu_quota_insufficient"
elif [[ "${gpu_ready}" != "true" ]]; then
  result="blocked"
  failure_reason="gpu_quota_insufficient"
elif [[ "${instance_ready}" != "true" ]]; then
  result="blocked"
  failure_reason="instance_quota_insufficient"
elif [[ "${disk_ready}" != "true" ]]; then
  result="blocked"
  failure_reason="disk_quota_insufficient"
fi

jq -n \
  --arg schema_version "psion.google_training_quota_preflight.v1" \
  --arg checked_at_utc "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  --arg project_id "${PROJECT_ID}" \
  --arg region "${region}" \
  --arg zone "${ZONE}" \
  --arg profile_id "${PROFILE_ID}" \
  --arg machine_type "${machine_type}" \
  --arg accelerator_type "${accelerator_type}" \
  --arg machine_type_state "${machine_type_state}" \
  --arg accelerator_state "${accelerator_state}" \
  --arg result "${result}" \
  --arg failure_reason "${failure_reason}" \
  --arg budget_topic "${topic_name}" \
  --arg capacity_failure_result "${capacity_failure_result}" \
  --argjson required_vcpus "${required_vcpus}" \
  --argjson required_memory_mb "${required_memory_mb}" \
  --argjson accelerator_count "${accelerator_count}" \
  --argjson boot_disk_gb "${boot_disk_gb}" \
  --argjson max_runtime_hours "${max_runtime_hours}" \
  --argjson declared_run_cost_ceiling_usd "${declared_run_cost_ceiling_usd}" \
  --argjson budget_amount_usd "${budget_amount_usd}" \
  --argjson max_launch_attempts "${max_launch_attempts}" \
  --argjson cpu_quota_metrics "${cpu_quota_metrics_json}" \
  --argjson gpu_quota_metrics "${gpu_quota_metrics_json}" \
  --argjson cpu_candidates "${cpu_candidates_json}" \
  --argjson gpu_candidates "${gpu_candidates_json}" \
  --argjson cpu_summary "${cpu_summary}" \
  --argjson gpu_summary "${gpu_summary}" \
  --argjson instance_summary "${instance_summary}" \
  --argjson disk_summary "${disk_summary}" \
  --argjson zone_fallback_order "$(jq '.zone_fallback_order' <<<"${profile_json}")" \
  '{
    schema_version: $schema_version,
    checked_at_utc: $checked_at_utc,
    project_id: $project_id,
    region: $region,
    zone: $zone,
    profile_id: $profile_id,
    machine_type: $machine_type,
    accelerator_type: $accelerator_type,
    required_vcpus: $required_vcpus,
    required_memory_mb: $required_memory_mb,
    accelerator_count: $accelerator_count,
    boot_disk_gb: $boot_disk_gb,
    max_runtime_hours: $max_runtime_hours,
    declared_run_cost_ceiling_usd: $declared_run_cost_ceiling_usd,
    monthly_budget_amount_usd: $budget_amount_usd,
    budget_notifications_topic: $budget_topic,
    max_launch_attempts: $max_launch_attempts,
    capacity_failure_result: $capacity_failure_result,
    zone_fallback_order: $zone_fallback_order,
    machine_type_state: $machine_type_state,
    accelerator_state: $accelerator_state,
    quotas: {
      cpu_metrics: $cpu_quota_metrics,
      accelerator_metrics: $gpu_quota_metrics,
      cpu_candidates: $cpu_candidates,
      accelerator_candidates: $gpu_candidates,
      cpu: $cpu_summary,
      accelerator: $gpu_summary,
      instances: $instance_summary,
      disk: $disk_summary
    },
    result: $result,
    failure_reason: (if $failure_reason == "" then null else $failure_reason end)
  }'

if [[ "${result}" != "ready" ]]; then
  exit 1
fi

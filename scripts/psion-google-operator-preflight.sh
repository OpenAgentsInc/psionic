#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
POLICY_FILE="${POLICY_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_operator_preflight_policy_v1.json}"
IDENTITY_PROFILE_FILE="${IDENTITY_PROFILE_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_training_identity_profile_v1.json}"
QUOTA_PREFLIGHT="${QUOTA_PREFLIGHT:-${REPO_ROOT}/scripts/psion-google-quota-preflight.sh}"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

PROJECT_ID="${PROJECT_ID:-$(jq -r '.project_id' "${POLICY_FILE}")}"
PROFILE_ID="${PROFILE_ID:-g2_l4_single_node_accelerated}"
ZONE="${ZONE:-}"

usage() {
  cat <<'EOF'
Usage: psion-google-operator-preflight.sh [options]

Options:
  --profile <profile_id>      Launch profile to validate.
  --zone <zone>               Force one zone instead of the profile default.
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

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

version_ge() {
  local actual="$1"
  local minimum="$2"
  [[ "$(printf '%s\n%s\n' "${minimum}" "${actual}" | sort -V | head -n 1)" == "${minimum}" ]]
}

tool_versions_json="$(gcloud version --format=json)"
gcloud_version="$(jq -r '.["Google Cloud SDK"]' <<<"${tool_versions_json}")"
bq_version="$(jq -r '.bq' <<<"${tool_versions_json}")"

errors_file="$(mktemp)"
trap 'rm -f "${errors_file}"' EXIT

record_error() {
  local check_name="$1"
  local detail="$2"
  jq -nc --arg check_name "${check_name}" --arg detail "${detail}" \
    '{check_name: $check_name, detail: $detail}' >> "${errors_file}"
}

while IFS= read -r required_command; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    record_error "command" "required command missing: ${required_command}"
  fi
done < <(jq -r '.required_commands[]' "${POLICY_FILE}")

minimum_gcloud="$(jq -r '.minimum_tool_versions.gcloud' "${POLICY_FILE}")"
minimum_bq="$(jq -r '.minimum_tool_versions.bq' "${POLICY_FILE}")"
if ! version_ge "${gcloud_version}" "${minimum_gcloud}"; then
  record_error "gcloud_version" "gcloud ${gcloud_version} is below required minimum ${minimum_gcloud}"
fi
if ! version_ge "${bq_version}" "${minimum_bq}"; then
  record_error "bq_version" "bq ${bq_version} is below required minimum ${minimum_bq}"
fi

policy_project_id="$(jq -r '.project_id' "${POLICY_FILE}")"
if [[ "${PROJECT_ID}" != "${policy_project_id}" ]]; then
  record_error "project_override" "unsupported project override ${PROJECT_ID}; this runbook is pinned to ${policy_project_id}"
fi

supported_profiles_json="$(jq -c '.supported_profiles' "${POLICY_FILE}")"
if ! jq -e --arg profile_id "${PROFILE_ID}" 'index($profile_id) != null' <<<"${supported_profiles_json}" >/dev/null; then
  record_error "profile" "unsupported profile ${PROFILE_ID}"
fi

active_account="$(gcloud config get-value account 2>/dev/null || true)"
active_project="$(gcloud config get-value project 2>/dev/null || true)"
if [[ -z "${active_account}" ]]; then
  record_error "auth_account" "no active gcloud account configured"
fi
if [[ "${active_project}" != "${PROJECT_ID}" ]]; then
  record_error "auth_project" "active gcloud project ${active_project:-<none>} does not match required project ${PROJECT_ID}"
fi
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  record_error "access_token" "failed to mint an access token from the active gcloud auth context"
fi

bucket_url="$(jq -r '.bucket_url' "${POLICY_FILE}")"
if ! gcloud storage ls "${bucket_url}" >/dev/null 2>&1; then
  record_error "bucket_access" "failed to list ${bucket_url}"
fi

finops_dataset="$(jq -r '.finops_dataset' "${POLICY_FILE}")"
finops_table="$(jq -r '.finops_price_profile_table' "${POLICY_FILE}")"
if ! bq show --format=prettyjson "${PROJECT_ID}:${finops_dataset}" >/dev/null 2>&1; then
  record_error "finops_dataset" "failed to read dataset ${PROJECT_ID}:${finops_dataset}"
fi
if ! bq show --format=prettyjson "${PROJECT_ID}:${finops_dataset}.${finops_table}" >/dev/null 2>&1; then
  record_error "finops_table" "failed to read table ${PROJECT_ID}:${finops_dataset}.${finops_table}"
fi

training_service_account="$(jq -r '.training_service_account' "${POLICY_FILE}")"
if ! gcloud iam service-accounts describe "${training_service_account}" --format='value(email)' >/dev/null 2>&1; then
  record_error "training_service_account" "failed to describe ${training_service_account}"
fi

if [[ -f "${IDENTITY_PROFILE_FILE}" ]]; then
  while IFS= read -r required_project_role; do
    if ! gcloud projects get-iam-policy "${PROJECT_ID}" \
      --flatten='bindings[].members' \
      --filter="bindings.members:serviceAccount:${training_service_account} AND bindings.role:${required_project_role}" \
      --format='value(bindings.role)' | grep -qx "${required_project_role}"; then
      record_error "training_service_account_project_role" "${training_service_account} is missing required project role ${required_project_role}"
    fi
  done < <(jq -r '.required_project_roles[]' "${IDENTITY_PROFILE_FILE}")
fi

if [[ -f "${IDENTITY_PROFILE_FILE}" ]]; then
  storage_bucket_url="$(jq -r '.bucket_url' "${IDENTITY_PROFILE_FILE}")"
  while IFS= read -r required_bucket_role; do
    if ! gcloud storage buckets get-iam-policy "${storage_bucket_url}" --format=json | \
      jq -e \
        --arg member "serviceAccount:${training_service_account}" \
        --arg role "${required_bucket_role}" \
        '.bindings[] | select(.role == $role and (.members[]? == $member))' >/dev/null; then
      record_error "training_service_account_bucket_role" "${training_service_account} is missing required bucket role ${required_bucket_role} on ${storage_bucket_url}"
    fi
  done < <(jq -r '.required_bucket_roles[]' "${IDENTITY_PROFILE_FILE}")
fi

quota_preflight_json=""
if quota_preflight_json="$(PROJECT_ID="${PROJECT_ID}" PROFILE_ID="${PROFILE_ID}" ZONE="${ZONE}" bash "${QUOTA_PREFLIGHT}" 2>/dev/null)"; then
  :
else
  quota_preflight_json="$(PROJECT_ID="${PROJECT_ID}" PROFILE_ID="${PROFILE_ID}" ZONE="${ZONE}" bash "${QUOTA_PREFLIGHT}" 2>/tmp/psion-google-operator-preflight-quota.err || true)"
  record_error "quota_preflight" "quota preflight failed for profile ${PROFILE_ID}${ZONE:+ in zone ${ZONE}}"
fi

if [[ -s "${errors_file}" ]]; then
  result="refused"
  errors_json="$(jq -s '.' "${errors_file}")"
else
  result="ready"
  errors_json='[]'
fi

jq -n \
  --arg schema_version "psion.google_operator_preflight.v1" \
  --arg checked_at_utc "$(timestamp_utc)" \
  --arg project_id "${PROJECT_ID}" \
  --arg profile_id "${PROFILE_ID}" \
  --arg zone "${ZONE}" \
  --arg active_account "${active_account}" \
  --arg active_project "${active_project}" \
  --arg gcloud_version "${gcloud_version}" \
  --arg bq_version "${bq_version}" \
  --arg minimum_gcloud "${minimum_gcloud}" \
  --arg minimum_bq "${minimum_bq}" \
  --arg bucket_url "${bucket_url}" \
  --arg finops_dataset "${finops_dataset}" \
  --arg finops_table "${finops_table}" \
  --arg training_service_account "${training_service_account}" \
  --arg result "${result}" \
  --argjson required_commands "$(jq '.required_commands' "${POLICY_FILE}")" \
  --argjson supported_profiles "$(jq '.supported_profiles' "${POLICY_FILE}")" \
  --argjson quota_preflight "${quota_preflight_json:-null}" \
  --argjson errors "${errors_json}" \
  '{
    schema_version: $schema_version,
    checked_at_utc: $checked_at_utc,
    project_id: $project_id,
    profile_id: $profile_id,
    zone_override: (if $zone == "" then null else $zone end),
    local_tooling: {
      required_commands: $required_commands,
      versions: {
        gcloud: $gcloud_version,
        bq: $bq_version
      },
      minimum_versions: {
        gcloud: $minimum_gcloud,
        bq: $minimum_bq
      }
    },
    auth_posture: {
      active_account: (if $active_account == "" then null else $active_account end),
      active_project: (if $active_project == "" then null else $active_project end),
      bucket_url: $bucket_url,
      finops_dataset: $finops_dataset,
      finops_price_profile_table: $finops_table,
      training_service_account: $training_service_account
    },
    secret_posture: {
      prefer_attached_service_account: true,
      runtime_secret_injection_refused_by_default: true,
      record_secret_dependencies_without_values: true
    },
    supported_profiles: $supported_profiles,
    quota_preflight: $quota_preflight,
    result: $result,
    errors: $errors
  }'

if [[ "${result}" != "ready" ]]; then
  exit 1
fi

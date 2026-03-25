#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
IDENTITY_PROFILE_FILE="${IDENTITY_PROFILE_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_two_node_swarm_identity_profile_v1.json}"
STORAGE_PROFILE_FILE="${STORAGE_PROFILE_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_training_storage_profile_v1.json}"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

PROJECT_ID="${PROJECT_ID:-$(jq -r '.project_id' "${IDENTITY_PROFILE_FILE}")}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-$(jq -r '.service_account_email' "${IDENTITY_PROFILE_FILE}")}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-${SERVICE_ACCOUNT_EMAIL%@*}}"
SERVICE_ACCOUNT_DISPLAY_NAME="${SERVICE_ACCOUNT_DISPLAY_NAME:-$(jq -r '.service_account_display_name' "${IDENTITY_PROFILE_FILE}")}"
SERVICE_ACCOUNT_DESCRIPTION="${SERVICE_ACCOUNT_DESCRIPTION:-$(jq -r '.service_account_description' "${IDENTITY_PROFILE_FILE}")}"
BUCKET_URL="${BUCKET_URL:-$(jq -r '.bucket_url' "${IDENTITY_PROFILE_FILE}")}"
ACTIVE_ACCOUNT="${ACTIVE_ACCOUNT:-$(gcloud config get-value core/account 2>/dev/null)}"

if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  :
else
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="${SERVICE_ACCOUNT_DISPLAY_NAME}" \
    --description="${SERVICE_ACCOUNT_DESCRIPTION}" >/dev/null
fi

for _ in $(seq 1 10); do
  if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if [[ -n "${ACTIVE_ACCOUNT}" ]]; then
  gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT_EMAIL}" \
    --project="${PROJECT_ID}" \
    --member="user:${ACTIVE_ACCOUNT}" \
    --role="roles/iam.serviceAccountTokenCreator" \
    --quiet >/dev/null
fi

while IFS= read -r role; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="${role}" \
    --quiet >/dev/null
done < <(jq -r '.required_project_roles[]' "${IDENTITY_PROFILE_FILE}")

while IFS= read -r role; do
  gcloud storage buckets add-iam-policy-binding "${BUCKET_URL}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="${role}" \
    --quiet >/dev/null
done < <(jq -r '.required_bucket_roles[]' "${IDENTITY_PROFILE_FILE}")

project_labels="$(jq -c '.project_label_updates' "${IDENTITY_PROFILE_FILE}")"
bucket_labels="$(jq -c '.bucket_label_updates' "${IDENTITY_PROFILE_FILE}")"
project_number="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
access_token="$(gcloud auth print-access-token)"
curl -sS -X PATCH \
  -H "Authorization: Bearer ${access_token}" \
  -H "Content-Type: application/json" \
  "https://cloudresourcemanager.googleapis.com/v3/projects/${project_number}?updateMask=labels" \
  -d "{\"labels\":${project_labels}}" >/dev/null

gcloud storage buckets update "${BUCKET_URL}" \
  --project="${PROJECT_ID}" \
  --update-labels="$(jq -r 'to_entries | map("\(.key)=\(.value)") | join(",")' <<<"${bucket_labels}")" >/dev/null

gcloud storage cp --quiet \
  "${IDENTITY_PROFILE_FILE}" \
  "${BUCKET_URL}/manifests/psion_google_two_node_swarm_identity_profile_v1.json" >/dev/null
gcloud storage cp --quiet \
  "${STORAGE_PROFILE_FILE}" \
  "${BUCKET_URL}/manifests/psion_google_training_storage_profile_v1.json" >/dev/null

jq -n \
  --arg project_id "${PROJECT_ID}" \
  --arg service_account_email "${SERVICE_ACCOUNT_EMAIL}" \
  --arg bucket_url "${BUCKET_URL}" \
  '{
    project_id: $project_id,
    service_account_email: $service_account_email,
    bucket_url: $bucket_url
  }'

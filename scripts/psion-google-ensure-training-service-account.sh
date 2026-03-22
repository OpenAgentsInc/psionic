#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
PROJECT_NUMBER="${PROJECT_NUMBER:-157437760789}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-psion-train-single-node}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"
BUCKET_URL="${BUCKET_URL:-gs://openagentsgemini-psion-train-us-central1}"
ACTIVE_ACCOUNT="${ACTIVE_ACCOUNT:-$(gcloud config get-value core/account 2>/dev/null)}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
IDENTITY_PROFILE_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_training_identity_profile_v1.json"
STORAGE_PROFILE_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_training_storage_profile_v1.json"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "service account already exists: ${SERVICE_ACCOUNT_EMAIL}"
else
  echo "creating service account: ${SERVICE_ACCOUNT_EMAIL}"
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="Psion single-node training runtime" \
    --description="Bounded Psion single-node Google training pilot runtime."
fi

for _ in $(seq 1 10); do
  if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "error: service account did not become visible in time: ${SERVICE_ACCOUNT_EMAIL}" >&2
  exit 1
fi

echo "granting operator impersonation to ${ACTIVE_ACCOUNT}"
gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT_EMAIL}" \
  --project="${PROJECT_ID}" \
  --member="user:${ACTIVE_ACCOUNT}" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --quiet >/dev/null

echo "granting project roles to ${SERVICE_ACCOUNT_EMAIL}"
for role in roles/logging.logWriter roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="${role}" \
    --quiet >/dev/null
done

echo "granting bucket roles to ${SERVICE_ACCOUNT_EMAIL}"
for role in roles/storage.objectAdmin roles/storage.legacyBucketReader; do
  applied_binding=0
  for _ in $(seq 1 10); do
    if gcloud storage buckets add-iam-policy-binding "${BUCKET_URL}" \
      --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
      --role="${role}" \
      --quiet >/dev/null 2>&1; then
      applied_binding=1
      break
    fi
    sleep 2
  done
  if [[ "${applied_binding}" -ne 1 ]]; then
    echo "error: failed to bind ${role} on ${BUCKET_URL} to ${SERVICE_ACCOUNT_EMAIL}" >&2
    exit 1
  fi
done

echo "updating project hygiene labels"
project_json="$(gcloud projects describe "${PROJECT_ID}" --format=json)"
merged_labels="$(
  jq -c '.labels + {"environment":"development","psion_lane":"google_single_node"}' <<<"${project_json}"
)"
access_token="$(gcloud auth print-access-token)"
curl -sS -X PATCH \
  -H "Authorization: Bearer ${access_token}" \
  -H "Content-Type: application/json" \
  "https://cloudresourcemanager.googleapis.com/v3/projects/${PROJECT_NUMBER}?updateMask=labels" \
  -d "{\"labels\":${merged_labels}}" >/dev/null

echo "updating bucket labels"
gcloud storage buckets update "${BUCKET_URL}" \
  --project="${PROJECT_ID}" \
  --update-labels=environment=development,lane=psion,profile=single-node,purpose=train >/dev/null

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

cp "${IDENTITY_PROFILE_FILE}" "${tmpdir}/psion_google_training_identity_profile_v1.json"
cp "${STORAGE_PROFILE_FILE}" "${tmpdir}/psion_google_training_storage_profile_v1.json"

gcloud storage cp --quiet \
  "${tmpdir}/psion_google_training_identity_profile_v1.json" \
  "${BUCKET_URL}/manifests/psion_google_training_identity_profile_v1.json" >/dev/null
gcloud storage cp --quiet \
  "${tmpdir}/psion_google_training_storage_profile_v1.json" \
  "${BUCKET_URL}/manifests/psion_google_training_storage_profile_v1.json" >/dev/null

echo "identity summary:"
gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" --format=json | \
  jq '{email, displayName, description, uniqueId}'

echo "project labels:"
gcloud projects describe "${PROJECT_ID}" --format=json | jq '{projectId, labels}'

echo "bucket labels:"
gcloud storage buckets describe "${BUCKET_URL}" --project="${PROJECT_ID}" --format=json | \
  jq '{name, labels}'

echo "project role bindings:"
gcloud projects get-iam-policy "${PROJECT_ID}" \
  --flatten='bindings[].members' \
  --filter="bindings.members:serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --format='table(bindings.role)'

echo "bucket role bindings:"
gcloud storage buckets get-iam-policy "${BUCKET_URL}" --format=json | \
  jq --arg member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" '
    [
      .bindings[]
      | select(.members[]? == $member)
      | .role
    ]'

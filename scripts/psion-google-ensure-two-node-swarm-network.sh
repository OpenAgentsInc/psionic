#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
NETWORK_POSTURE_FILE="${NETWORK_POSTURE_FILE:-${REPO_ROOT}/fixtures/psion/google/psion_google_two_node_swarm_network_posture_v1.json}"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

PROJECT_ID="${PROJECT_ID:-$(jq -r '.project_id' "${NETWORK_POSTURE_FILE}")}"
REGION="${REGION:-$(jq -r '.region' "${NETWORK_POSTURE_FILE}")}"
NETWORK="${NETWORK:-$(jq -r '.network' "${NETWORK_POSTURE_FILE}")}"
BUCKET_URL="${BUCKET_URL:-gs://openagentsgemini-psion-train-us-central1}"
ROUTER="${ROUTER:-$(jq -r '.egress_posture.router' "${NETWORK_POSTURE_FILE}")}"
EXPECTED_NAT="${EXPECTED_NAT:-$(jq -r '.egress_posture.nat' "${NETWORK_POSTURE_FILE}")}"
SSH_FIREWALL_RULE="${SSH_FIREWALL_RULE:-$(jq -r '.ssh_posture.firewall_rule_name' "${NETWORK_POSTURE_FILE}")}"
CLUSTER_FIREWALL_RULE="${CLUSTER_FIREWALL_RULE:-$(jq -r '.cluster_transport_posture.firewall_rule_name' "${NETWORK_POSTURE_FILE}")}"

join_csv() {
  jq -r 'join(",")'
}

ensure_subnetwork() {
  local name="$1"
  local cidr="$2"
  if gcloud compute networks subnets describe "${name}" --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
    local actual_range
    actual_range="$(
      gcloud compute networks subnets describe "${name}" \
        --project="${PROJECT_ID}" \
        --region="${REGION}" \
        --format='value(ipCidrRange)'
    )"
    if [[ "${actual_range}" != "${cidr}" ]]; then
      echo "error: subnetwork ${name} exists with ${actual_range}, expected ${cidr}" >&2
      exit 1
    fi
  else
    gcloud compute networks subnets create "${name}" \
      --project="${PROJECT_ID}" \
      --network="${NETWORK}" \
      --region="${REGION}" \
      --range="${cidr}" >/dev/null
  fi
}

ensure_firewall_rule() {
  local name="$1"
  local allow="$2"
  local source_ranges="$3"
  local target_tags="$4"
  local description="$5"
  if gcloud compute firewall-rules describe "${name}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud compute firewall-rules update "${name}" \
      --project="${PROJECT_ID}" \
      --allow="${allow}" \
      --source-ranges="${source_ranges}" \
      --target-tags="${target_tags}" >/dev/null
  else
    gcloud compute firewall-rules create "${name}" \
      --project="${PROJECT_ID}" \
      --network="${NETWORK}" \
      --direction=INGRESS \
      --priority=1000 \
      --allow="${allow}" \
      --source-ranges="${source_ranges}" \
      --target-tags="${target_tags}" \
      --description="${description}" >/dev/null
  fi
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

router_json="$(
  gcloud compute routers describe "${ROUTER}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format=json
)"
router_network="$(jq -r '.network | split("/") | last' <<<"${router_json}")"
nat_name="$(jq -r '.nats[0].name // empty' <<<"${router_json}")"
if [[ "${router_network}" != "${NETWORK}" ]]; then
  echo "error: router ${ROUTER} is attached to ${router_network}, expected ${NETWORK}" >&2
  exit 1
fi
if [[ "${nat_name}" != "${EXPECTED_NAT}" ]]; then
  echo "error: router ${ROUTER} is missing expected NAT ${EXPECTED_NAT}" >&2
  exit 1
fi

while IFS= read -r subnetwork_json; do
  name="$(jq -r '.subnetwork_name' <<<"${subnetwork_json}")"
  cidr="$(jq -r '.ip_cidr_range' <<<"${subnetwork_json}")"
  ensure_subnetwork "${name}" "${cidr}"
done < <(jq -c '.subnetworks[]' "${NETWORK_POSTURE_FILE}")

ssh_source_ranges="$(jq -c '.ssh_posture.source_ranges' "${NETWORK_POSTURE_FILE}" | join_csv)"
ssh_target_tags="$(jq -c '.ssh_posture.target_tags' "${NETWORK_POSTURE_FILE}" | join_csv)"
ssh_allowed="$(jq -c '.ssh_posture.allowed' "${NETWORK_POSTURE_FILE}" | join_csv)"
ensure_firewall_rule \
  "${SSH_FIREWALL_RULE}" \
  "${ssh_allowed}" \
  "${ssh_source_ranges}" \
  "${ssh_target_tags}" \
  "IAP SSH for Psion Google two-node swarm hosts."

cluster_source_ranges="$(jq -c '.cluster_transport_posture.source_ranges' "${NETWORK_POSTURE_FILE}" | join_csv)"
cluster_target_tags="$(jq -c '.cluster_transport_posture.target_tags' "${NETWORK_POSTURE_FILE}" | join_csv)"
cluster_allowed="$(jq -c '.cluster_transport_posture.allowed' "${NETWORK_POSTURE_FILE}" | join_csv)"
ensure_firewall_rule \
  "${CLUSTER_FIREWALL_RULE}" \
  "${cluster_allowed}" \
  "${cluster_source_ranges}" \
  "${cluster_target_tags}" \
  "Configured-peer cluster ingress for Psion Google two-node swarm hosts."

manifest_uri="${BUCKET_URL}/manifests/psion_google_two_node_swarm_network_posture_v1.json"
gcloud storage cp --quiet "${NETWORK_POSTURE_FILE}" "${manifest_uri}" >/dev/null
wait_for_object "${manifest_uri}"

jq -n \
  --arg project_id "${PROJECT_ID}" \
  --arg region "${REGION}" \
  --arg network "${NETWORK}" \
  --arg ssh_firewall_rule "${SSH_FIREWALL_RULE}" \
  --arg cluster_firewall_rule "${CLUSTER_FIREWALL_RULE}" \
  --arg manifest_uri "${manifest_uri}" \
  --argjson subnetworks "$(jq '.subnetworks' "${NETWORK_POSTURE_FILE}")" \
  '{
    project_id: $project_id,
    region: $region,
    network: $network,
    subnetworks: $subnetworks,
    ssh_firewall_rule: $ssh_firewall_rule,
    cluster_firewall_rule: $cluster_firewall_rule,
    manifest_uri: $manifest_uri
  }'

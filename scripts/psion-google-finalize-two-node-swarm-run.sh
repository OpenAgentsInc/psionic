#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
BUCKET_URL="${BUCKET_URL:-gs://openagentsgemini-psion-train-us-central1}"
RUN_ID="${RUN_ID:-}"
MANIFEST_URI="${MANIFEST_URI:-}"
COORDINATOR_IMPAIRMENT_RECEIPT="${COORDINATOR_IMPAIRMENT_RECEIPT:-}"
CONTRIBUTOR_IMPAIRMENT_RECEIPT="${CONTRIBUTOR_IMPAIRMENT_RECEIPT:-}"
BUNDLE_OUT="${BUNDLE_OUT:-}"
FINAL_MANIFEST_OUT="${FINAL_MANIFEST_OUT:-}"

usage() {
  cat <<'EOF'
Usage: psion-google-finalize-two-node-swarm-run.sh [options]

Options:
  --run-id <run_id>                           Resolve the cluster manifest from the training bucket.
  --manifest-uri <uri>                        Use one explicit cluster manifest object.
  --coordinator-impairment-receipt <path|gs://uri>
                                              Optional coordinator impairment receipt to upload or reuse.
  --contributor-impairment-receipt <path|gs://uri>
                                              Optional contributor impairment receipt to upload or reuse.
  --bundle-out <path>                         Also write the final evidence bundle locally.
  --final-manifest-out <path>                 Also write the final manifest locally.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --manifest-uri)
      MANIFEST_URI="$2"
      shift 2
      ;;
    --coordinator-impairment-receipt)
      COORDINATOR_IMPAIRMENT_RECEIPT="$2"
      shift 2
      ;;
    --contributor-impairment-receipt)
      CONTRIBUTOR_IMPAIRMENT_RECEIPT="$2"
      shift 2
      ;;
    --bundle-out)
      BUNDLE_OUT="$2"
      shift 2
      ;;
    --final-manifest-out)
      FINAL_MANIFEST_OUT="$2"
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

if [[ -z "${RUN_ID}" && -z "${MANIFEST_URI}" ]]; then
  echo "error: provide --run-id or --manifest-uri" >&2
  exit 1
fi

for required_command in gcloud jq python3; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "error: ${required_command} is required" >&2
    exit 1
  fi
done

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

compute_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

wait_for_object() {
  local object_path="$1"
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if gcloud storage ls "${object_path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: object ${object_path} did not become visible" >&2
  exit 1
}

copy_remote_to_local() {
  local remote_uri="$1"
  local local_path="$2"
  gcloud storage cp --quiet "${remote_uri}" "${local_path}" >/dev/null
}

append_artifact_record() {
  local artifact_kind="$1"
  local evidence_role="$2"
  local remote_uri="$3"
  local local_path="$4"
  local source_mode="$5"
  jq -nc \
    --arg artifact_kind "${artifact_kind}" \
    --arg evidence_role "${evidence_role}" \
    --arg remote_uri "${remote_uri}" \
    --arg sha256 "$(compute_sha256 "${local_path}")" \
    --arg byte_length "$(wc -c < "${local_path}" | tr -d ' ')" \
    --arg source_mode "${source_mode}" \
    '{
      artifact_kind: $artifact_kind,
      evidence_role: $evidence_role,
      remote_uri: $remote_uri,
      sha256: $sha256,
      byte_length: ($byte_length | tonumber),
      source_mode: $source_mode
    }' >> "${artifact_records_file}"
}

stage_optional_receipt() {
  local role="$1"
  local input="$2"
  local local_path="$3"
  local remote_uri="$4"
  if [[ -z "${input}" ]]; then
    return 1
  fi
  if [[ "${input}" == gs://* ]]; then
    copy_remote_to_local "${input}" "${local_path}"
    append_artifact_record "${role}_impairment_receipt" "receipt" "${input}" "${local_path}" "remote_existing"
  else
    cp "${input}" "${local_path}"
    gcloud storage cp --quiet "${local_path}" "${remote_uri}" >/dev/null
    wait_for_object "${remote_uri}"
    append_artifact_record "${role}_impairment_receipt" "receipt" "${remote_uri}" "${local_path}" "local_upload"
  fi
}

if [[ -z "${MANIFEST_URI}" ]]; then
  MANIFEST_URI="${BUCKET_URL}/runs/${RUN_ID}/launch/psion_google_two_node_swarm_cluster_manifest.json"
fi
wait_for_object "${MANIFEST_URI}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
artifact_records_file="${tmpdir}/artifact_records.jsonl"
: > "${artifact_records_file}"

manifest_file="${tmpdir}/cluster_manifest.json"
copy_remote_to_local "${MANIFEST_URI}" "${manifest_file}"
append_artifact_record "cluster_manifest" "manifest" "${MANIFEST_URI}" "${manifest_file}" "remote_existing"

RUN_ID="${RUN_ID:-$(jq -r '.run_id' "${manifest_file}")}"
run_prefix="$(jq -r '.run_prefix' "${manifest_file}")"
launch_receipt_uri="$(jq -r '.launch_receipt_uri' "${manifest_file}")"
final_manifest_uri="$(jq -r '.final_manifest_uri' "${manifest_file}")"
cluster_id="$(jq -r '.cluster_id' "${manifest_file}")"
selected_impairment_profile_id="$(jq -r '.selected_impairment_profile_id' "${manifest_file}")"
selected_zone_pair_id="$(jq -r '.selected_zone_pair_id' "${manifest_file}")"
contract_digest="$(jq -r '.contract_digest' "${manifest_file}")"

launch_receipt_file="${tmpdir}/launch_receipt.json"
copy_remote_to_local "${launch_receipt_uri}" "${launch_receipt_file}"
append_artifact_record "launch_receipt" "receipt" "${launch_receipt_uri}" "${launch_receipt_file}" "remote_existing"

coordinator_node_json="$(jq -c '.nodes[] | select(.role_kind == "coordinator_validator_aggregator_contributor")' "${manifest_file}")"
contributor_node_json="$(jq -c '.nodes[] | select(.role_kind == "contributor")' "${manifest_file}")"
if [[ -z "${coordinator_node_json}" || -z "${contributor_node_json}" ]]; then
  echo "error: cluster manifest did not contain both admitted node roles" >&2
  exit 1
fi

coordinator_bringup_uri="$(jq -r '.bringup_report_uri' <<<"${coordinator_node_json}")"
contributor_bringup_uri="$(jq -r '.bringup_report_uri' <<<"${contributor_node_json}")"
coordinator_runtime_uri="$(jq -r '.runtime_report_uri' <<<"${coordinator_node_json}")"
contributor_runtime_uri="$(jq -r '.runtime_report_uri' <<<"${contributor_node_json}")"

coordinator_bringup_file="${tmpdir}/coordinator_bringup.json"
contributor_bringup_file="${tmpdir}/contributor_bringup.json"
coordinator_runtime_file="${tmpdir}/coordinator_runtime.json"
contributor_runtime_file="${tmpdir}/contributor_runtime.json"

wait_for_object "${coordinator_bringup_uri}"
wait_for_object "${contributor_bringup_uri}"
wait_for_object "${coordinator_runtime_uri}"
wait_for_object "${contributor_runtime_uri}"

copy_remote_to_local "${coordinator_bringup_uri}" "${coordinator_bringup_file}"
copy_remote_to_local "${contributor_bringup_uri}" "${contributor_bringup_file}"
copy_remote_to_local "${coordinator_runtime_uri}" "${coordinator_runtime_file}"
copy_remote_to_local "${contributor_runtime_uri}" "${contributor_runtime_file}"

append_artifact_record "coordinator_bringup_report" "manifest" "${coordinator_bringup_uri}" "${coordinator_bringup_file}" "remote_existing"
append_artifact_record "contributor_bringup_report" "manifest" "${contributor_bringup_uri}" "${contributor_bringup_file}" "remote_existing"
append_artifact_record "coordinator_runtime_report" "manifest" "${coordinator_runtime_uri}" "${coordinator_runtime_file}" "remote_existing"
append_artifact_record "contributor_runtime_report" "manifest" "${contributor_runtime_uri}" "${contributor_runtime_file}" "remote_existing"

coordinator_impairment_file="${tmpdir}/coordinator_impairment.json"
contributor_impairment_file="${tmpdir}/contributor_impairment.json"
coordinator_impairment_uri="${run_prefix}/host/coordinator/psion_google_two_node_swarm_impairment_receipt.json"
contributor_impairment_uri="${run_prefix}/host/contributor/psion_google_two_node_swarm_impairment_receipt.json"
have_coordinator_impairment=false
have_contributor_impairment=false
if stage_optional_receipt "coordinator" "${COORDINATOR_IMPAIRMENT_RECEIPT}" "${coordinator_impairment_file}" "${coordinator_impairment_uri}"; then
  have_coordinator_impairment=true
fi
if stage_optional_receipt "contributor" "${CONTRIBUTOR_IMPAIRMENT_RECEIPT}" "${contributor_impairment_file}" "${contributor_impairment_uri}"; then
  have_contributor_impairment=true
fi

result_classification="bounded_success"
failure_detail=""
coordinator_bringup_status="$(jq -r '.status' "${coordinator_bringup_file}")"
contributor_bringup_status="$(jq -r '.status' "${contributor_bringup_file}")"
coordinator_runtime_role="$(jq -r '.runtime_role' "${coordinator_runtime_file}")"
contributor_runtime_role="$(jq -r '.runtime_role' "${contributor_runtime_file}")"
coordinator_submission_count="$(jq -r '.submission_receipts | length' "${coordinator_runtime_file}")"
validator_present="$(jq -r 'if .validator_summary == null then "false" else "true" end' "${coordinator_runtime_file}")"
promotion_present="$(jq -r 'if .promotion_receipt == null then "false" else "true" end' "${coordinator_runtime_file}")"
contributor_local_present="$(jq -r 'if .local_contribution == null then "false" else "true" end' "${contributor_runtime_file}")"
coordinator_profile_match="$(jq -r --arg profile_id "${selected_impairment_profile_id}" 'if .selected_impairment_profile_id == $profile_id then "true" else "false" end' "${coordinator_runtime_file}")"
contributor_profile_match="$(jq -r --arg profile_id "${selected_impairment_profile_id}" 'if .selected_impairment_profile_id == $profile_id then "true" else "false" end' "${contributor_runtime_file}")"

if [[ "${coordinator_bringup_status}" != "ready" || "${contributor_bringup_status}" != "ready" ]]; then
  result_classification="configured_peer_launch_failure"
  failure_detail="One or both node bring-up reports refused the admitted machine contract."
elif [[ "${coordinator_runtime_role}" != "coordinator" || "${contributor_runtime_role}" != "contributor" ]]; then
  result_classification="cluster_membership_failure"
  failure_detail="The retained runtime reports did not preserve the admitted coordinator and contributor role split."
elif [[ "${coordinator_profile_match}" != "true" || "${contributor_profile_match}" != "true" ]]; then
  result_classification="network_impairment_gate_failure"
  failure_detail="The retained runtime reports drifted from the selected impairment profile id in the launch manifest."
elif [[ "${selected_impairment_profile_id}" != "clean_baseline" && ( "${have_coordinator_impairment}" != "true" || "${have_contributor_impairment}" != "true" ) ]]; then
  result_classification="network_impairment_gate_failure"
  failure_detail="The run selected a non-clean impairment profile but the retained impairment receipts are incomplete."
elif [[ "${contributor_local_present}" != "true" || "${coordinator_submission_count}" -lt 2 ]]; then
  result_classification="contributor_execution_failure"
  failure_detail="The contributor did not retain one full local contribution plus two coordinator submission receipts."
elif [[ "${validator_present}" != "true" ]]; then
  result_classification="validator_refusal"
  failure_detail="The coordinator runtime report did not retain a validator summary."
elif [[ "${promotion_present}" != "true" ]]; then
  result_classification="aggregation_failure"
  failure_detail="The coordinator runtime report did not retain an aggregation or promotion receipt."
fi

artifacts_json="$(jq -s '.' "${artifact_records_file}")"
coordinator_impairment_json='null'
contributor_impairment_json='null'
if [[ "${have_coordinator_impairment}" == "true" ]]; then
  coordinator_impairment_json="$(cat "${coordinator_impairment_file}")"
fi
if [[ "${have_contributor_impairment}" == "true" ]]; then
  contributor_impairment_json="$(cat "${contributor_impairment_file}")"
fi

evidence_bundle_file="${tmpdir}/psion_google_two_node_swarm_evidence_bundle.json"
evidence_bundle_uri="${run_prefix}/final/psion_google_two_node_swarm_evidence_bundle.json"
jq -n \
  --arg schema_version "psion.google_two_node_swarm_evidence_bundle.v1" \
  --arg created_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --arg cluster_id "${cluster_id}" \
  --arg contract_digest "${contract_digest}" \
  --arg selected_zone_pair_id "${selected_zone_pair_id}" \
  --arg selected_impairment_profile_id "${selected_impairment_profile_id}" \
  --arg result_classification "${result_classification}" \
  --arg failure_detail "${failure_detail}" \
  --arg manifest_uri "${MANIFEST_URI}" \
  --arg launch_receipt_uri "${launch_receipt_uri}" \
  --argjson coordinator_node "${coordinator_node_json}" \
  --argjson contributor_node "${contributor_node_json}" \
  --argjson coordinator_bringup_report "$(cat "${coordinator_bringup_file}")" \
  --argjson contributor_bringup_report "$(cat "${contributor_bringup_file}")" \
  --argjson coordinator_runtime_report "$(cat "${coordinator_runtime_file}")" \
  --argjson contributor_runtime_report "$(cat "${contributor_runtime_file}")" \
  --argjson coordinator_impairment_receipt "${coordinator_impairment_json}" \
  --argjson contributor_impairment_receipt "${contributor_impairment_json}" \
  --argjson retained_objects "${artifacts_json}" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    cluster_id: $cluster_id,
    contract_digest: $contract_digest,
    selected_zone_pair_id: $selected_zone_pair_id,
    selected_impairment_profile_id: $selected_impairment_profile_id,
    result_classification: $result_classification,
    failure_detail: (if $failure_detail == "" then null else $failure_detail end),
    topology: {
      cluster_manifest_uri: $manifest_uri,
      cluster_manifest_sha256: ($retained_objects[] | select(.artifact_kind == "cluster_manifest") | .sha256),
      launch_receipt_uri: $launch_receipt_uri,
      launch_receipt_sha256: ($retained_objects[] | select(.artifact_kind == "launch_receipt") | .sha256),
      nodes: [$coordinator_node, $contributor_node]
    },
    bringup_reports: [
      $coordinator_bringup_report,
      $contributor_bringup_report
    ],
    impairment_receipts: [
      $coordinator_impairment_receipt,
      $contributor_impairment_receipt
    ] | map(select(. != null)),
    runtime_reports: [
      $coordinator_runtime_report,
      $contributor_runtime_report
    ],
    validator_posture: $coordinator_runtime_report.validator_summary,
    aggregation_posture: $coordinator_runtime_report.promotion_receipt,
    cluster_health: {
      heartbeat_receipt_count: ($coordinator_runtime_report.heartbeat_receipts | length),
      acknowledgement_receipt_count: ($coordinator_runtime_report.acknowledgement_receipts | length),
      submission_receipt_count: ($coordinator_runtime_report.submission_receipts | length),
      coordinator_backend_label: $coordinator_runtime_report.execution_backend_label,
      contributor_backend_label: $contributor_runtime_report.execution_backend_label
    },
    retained_objects: $retained_objects,
    claim_boundary: "This evidence bundle covers one bounded Google two-node configured-peer adapter-delta lane. It binds launch truth, node-local bring-up truth, optional cluster-port impairment receipts, coordinator and contributor runtime reports, validator posture, aggregation posture, and final typed result classification. It does not claim trusted-cluster full-model training, public discovery, or broader Google swarm completion.",
    detail: "The bundle preserves the exact two-node Google swarm artifacts needed to audit topology, host readiness, impairment posture, contribution flow, validator posture, aggregation posture, and final disposition after the VMs are gone."
  }' > "${evidence_bundle_file}"

gcloud storage cp --quiet "${evidence_bundle_file}" "${evidence_bundle_uri}" >/dev/null
wait_for_object "${evidence_bundle_uri}"
append_artifact_record "cluster_evidence_bundle" "manifest" "${evidence_bundle_uri}" "${evidence_bundle_file}" "local_upload"
artifacts_json="$(jq -s '.' "${artifact_records_file}")"

final_manifest_file="${tmpdir}/psion_google_two_node_swarm_final_manifest.json"
jq -n \
  --arg schema_version "psion.google_two_node_swarm_final_manifest.v1" \
  --arg created_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --arg cluster_id "${cluster_id}" \
  --arg result_classification "${result_classification}" \
  --arg failure_detail "${failure_detail}" \
  --arg manifest_uri "${MANIFEST_URI}" \
  --arg launch_receipt_uri "${launch_receipt_uri}" \
  --arg evidence_bundle_uri "${evidence_bundle_uri}" \
  --arg evidence_bundle_sha256 "$(compute_sha256 "${evidence_bundle_file}")" \
  --arg selected_impairment_profile_id "${selected_impairment_profile_id}" \
  --argjson retained_objects "${artifacts_json}" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    cluster_id: $cluster_id,
    result_classification: $result_classification,
    failure_detail: (if $failure_detail == "" then null else $failure_detail end),
    selected_impairment_profile_id: $selected_impairment_profile_id,
    launch_artifacts: {
      cluster_manifest_uri: $manifest_uri,
      launch_receipt_uri: $launch_receipt_uri
    },
    evidence_bundle: {
      remote_uri: $evidence_bundle_uri,
      sha256: $evidence_bundle_sha256
    },
    retained_objects: $retained_objects,
    detail: "The final manifest binds the Google two-node swarm launch artifacts to the uploaded cluster-wide evidence bundle and preserves the final typed result classification for teardown and audit."
  }' > "${final_manifest_file}"

gcloud storage cp --quiet "${final_manifest_file}" "${final_manifest_uri}" >/dev/null
wait_for_object "${final_manifest_uri}"

if [[ -n "${BUNDLE_OUT}" ]]; then
  mkdir -p "$(dirname "${BUNDLE_OUT}")"
  cp "${evidence_bundle_file}" "${BUNDLE_OUT}"
fi
if [[ -n "${FINAL_MANIFEST_OUT}" ]]; then
  mkdir -p "$(dirname "${FINAL_MANIFEST_OUT}")"
  cp "${final_manifest_file}" "${FINAL_MANIFEST_OUT}"
fi

echo "google two-node swarm final manifest:"
cat "${final_manifest_file}"

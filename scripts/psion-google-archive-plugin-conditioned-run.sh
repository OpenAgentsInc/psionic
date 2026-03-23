#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
POLICY_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_plugin_conditioned_archive_policy_v1.json"
RUN_OUTPUT_DIR="${REPO_ROOT}/target/psion_google_plugin_conditioned_run"
FILE_STEM=""
MANIFEST_OUT="${PSION_ARCHIVE_MANIFEST_OUT:-}"

usage() {
  cat <<'EOF'
Usage: psion-google-archive-plugin-conditioned-run.sh [options] [run_output_dir]

Options:
  --stem <file_stem>         Required file stem, for example psion_plugin_host_native_reference.
  --manifest-out <path>      Write the generated archive manifest to one local path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stem)
      FILE_STEM="$2"
      shift 2
      ;;
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      RUN_OUTPUT_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "${FILE_STEM}" ]]; then
  echo "error: --stem is required" >&2
  usage >&2
  exit 1
fi

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

sanitize_component() {
  sed -E 's/[^A-Za-z0-9.-]+/-/g; s/^-+//; s/-+$//' <<<"$1"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "error: required file not found: ${path}" >&2
    exit 1
  fi
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

if [[ ! -d "${RUN_OUTPUT_DIR}" ]]; then
  echo "error: run output directory not found: ${RUN_OUTPUT_DIR}" >&2
  exit 1
fi

run_bundle_file="${RUN_OUTPUT_DIR}/${FILE_STEM}_run_bundle.json"
stage_bundle_file="${RUN_OUTPUT_DIR}/${FILE_STEM}_stage_bundle.json"
stage_receipt_file="${RUN_OUTPUT_DIR}/${FILE_STEM}_stage_receipt.json"
model_artifact_file="${RUN_OUTPUT_DIR}/${FILE_STEM}_model_artifact.json"
evaluation_receipt_file="${RUN_OUTPUT_DIR}/${FILE_STEM}_evaluation_receipt.json"
checkpoint_evidence_file="${RUN_OUTPUT_DIR}/${FILE_STEM}_checkpoint_evidence.json"
run_summary_file="${RUN_OUTPUT_DIR}/${FILE_STEM}_run_summary.json"

require_file "${run_bundle_file}"
require_file "${stage_bundle_file}"
require_file "${stage_receipt_file}"
require_file "${model_artifact_file}"
require_file "${evaluation_receipt_file}"
require_file "${checkpoint_evidence_file}"
require_file "${run_summary_file}"

project_id="$(jq -r '.project_id' "${POLICY_FILE}")"
bucket_url="$(jq -r '.bucket_url' "${POLICY_FILE}")"
archive_prefix="$(jq -r '.checkpoint_archive_prefix' "${POLICY_FILE}")"
archive_manifest_name="$(jq -r '.archive_manifest_name' "${POLICY_FILE}")"
archive_mode="$(jq -r '.archive_mode' "${POLICY_FILE}")"
storage_profile_json="$(jq '.storage_profile' "${POLICY_FILE}")"
created_at_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
git_revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
git_revision_short="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
input_package_descriptor_uri="${PSION_INPUT_PACKAGE_DESCRIPTOR_URI:-}"

run_id="$(jq -r '.run_id' "${checkpoint_evidence_file}")"
lane_id="$(jq -r '.lane_id' "${checkpoint_evidence_file}")"
checkpoint_family="$(jq -r '.checkpoint_family' "${checkpoint_evidence_file}")"
latest_checkpoint_ref="$(jq -r '.latest_checkpoint_ref' "${checkpoint_evidence_file}")"
latest_checkpoint_step="$(jq -r '.latest_checkpoint_step' "${checkpoint_evidence_file}")"
checkpoint_ref_count="$(jq -r '.checkpoint_ref_count' "${checkpoint_evidence_file}")"
stage_receipt_digest="$(jq -r '.stage_receipt_digest' "${checkpoint_evidence_file}")"
evaluation_receipt_digest="$(jq -r '.evaluation_receipt_digest' "${checkpoint_evidence_file}")"
model_artifact_digest="$(jq -r '.model_artifact_digest' "${checkpoint_evidence_file}")"

archive_id="psion-google-plugin-conditioned-${git_revision_short}-$(date -u '+%Y%m%dt%H%M%Sz' | tr '[:upper:]' '[:lower:]')"
lane_component="$(sanitize_component "${lane_id}")"
checkpoint_component="$(sanitize_component "${latest_checkpoint_ref}")"
checkpoint_prefix="${archive_prefix}/${lane_component}/${run_id}/${checkpoint_component}"
archive_manifest_uri="${checkpoint_prefix}/archive/${archive_manifest_name}"
artifact_records_file="${tmpdir}/artifact_records.jsonl"
: > "${artifact_records_file}"

upload_artifact() {
  local artifact_kind="$1"
  local local_path="$2"
  local remote_path="$3"
  local sha256
  sha256="$(compute_sha256 "${local_path}")"
  gcloud storage cp --quiet "${local_path}" "${remote_path}" >/dev/null
  wait_for_object "${remote_path}"
  jq -nc \
    --arg artifact_kind "${artifact_kind}" \
    --arg local_path "${local_path}" \
    --arg remote_uri "${remote_path}" \
    --arg sha256 "${sha256}" \
    --arg byte_length "$(wc -c < "${local_path}" | tr -d ' ')" \
    '{
      artifact_kind: $artifact_kind,
      local_path: $local_path,
      remote_uri: $remote_uri,
      sha256: $sha256,
      byte_length: ($byte_length | tonumber)
    }' >> "${artifact_records_file}"
}

upload_artifact "plugin_conditioned_run_bundle" "${run_bundle_file}" "${checkpoint_prefix}/bundles/$(basename "${run_bundle_file}")"
upload_artifact "plugin_conditioned_stage_bundle" "${stage_bundle_file}" "${checkpoint_prefix}/bundles/$(basename "${stage_bundle_file}")"
upload_artifact "plugin_conditioned_stage_receipt" "${stage_receipt_file}" "${checkpoint_prefix}/receipts/$(basename "${stage_receipt_file}")"
upload_artifact "plugin_conditioned_model_artifact" "${model_artifact_file}" "${checkpoint_prefix}/artifacts/$(basename "${model_artifact_file}")"
upload_artifact "plugin_conditioned_evaluation_receipt" "${evaluation_receipt_file}" "${checkpoint_prefix}/receipts/$(basename "${evaluation_receipt_file}")"
upload_artifact "plugin_conditioned_checkpoint_evidence" "${checkpoint_evidence_file}" "${checkpoint_prefix}/manifests/$(basename "${checkpoint_evidence_file}")"
upload_artifact "plugin_conditioned_run_summary" "${run_summary_file}" "${checkpoint_prefix}/manifests/$(basename "${run_summary_file}")"

artifacts_json="$(jq -s '.' "${artifact_records_file}")"
archive_manifest_file="${tmpdir}/${archive_manifest_name}"

jq -n \
  --arg schema_version "psion.google_plugin_conditioned_archive_manifest.v1" \
  --arg archive_id "${archive_id}" \
  --arg created_at_utc "${created_at_utc}" \
  --arg project_id "${project_id}" \
  --arg bucket_url "${bucket_url}" \
  --arg checkpoint_prefix "${checkpoint_prefix}" \
  --arg archive_manifest_uri "${archive_manifest_uri}" \
  --arg repo_git_revision "${git_revision}" \
  --arg input_package_descriptor_uri "${input_package_descriptor_uri}" \
  --arg run_id "${run_id}" \
  --arg lane_id "${lane_id}" \
  --arg checkpoint_family "${checkpoint_family}" \
  --arg latest_checkpoint_ref "${latest_checkpoint_ref}" \
  --argjson latest_checkpoint_step "${latest_checkpoint_step}" \
  --argjson checkpoint_ref_count "${checkpoint_ref_count}" \
  --arg stage_receipt_digest "${stage_receipt_digest}" \
  --arg evaluation_receipt_digest "${evaluation_receipt_digest}" \
  --arg model_artifact_digest "${model_artifact_digest}" \
  --arg archive_mode "${archive_mode}" \
  --argjson storage_profile "${storage_profile_json}" \
  --argjson artifacts "${artifacts_json}" \
  '{
    schema_version: $schema_version,
    archive_id: $archive_id,
    created_at_utc: $created_at_utc,
    project_id: $project_id,
    bucket_url: $bucket_url,
    checkpoint_prefix: $checkpoint_prefix,
    archive_manifest_uri: $archive_manifest_uri,
    repo_git_revision: $repo_git_revision,
    input_package_descriptor_uri: (if $input_package_descriptor_uri == "" then null else $input_package_descriptor_uri end),
    run_id: $run_id,
    lane_id: $lane_id,
    checkpoint_family: $checkpoint_family,
    latest_checkpoint_ref: $latest_checkpoint_ref,
    latest_checkpoint_step: $latest_checkpoint_step,
    checkpoint_ref_count: $checkpoint_ref_count,
    stage_receipt_digest: $stage_receipt_digest,
    evaluation_receipt_digest: $evaluation_receipt_digest,
    model_artifact_digest: $model_artifact_digest,
    archive_mode: $archive_mode,
    local_disk_authority: false,
    storage_profile: $storage_profile,
    artifacts: $artifacts,
    detail: "Plugin-conditioned Google archive stores the bounded run bundle, stage bundle, model artifact, evaluation receipt, and logical checkpoint evidence in GCS without implying a broader dense-checkpoint training lane than the current plugin-conditioned reference artifact actually proves."
  }' > "${archive_manifest_file}"

gcloud storage cp --quiet "${archive_manifest_file}" "${archive_manifest_uri}" >/dev/null
wait_for_object "${archive_manifest_uri}"

if [[ -n "${MANIFEST_OUT}" ]]; then
  cp "${archive_manifest_file}" "${MANIFEST_OUT}"
fi

echo "plugin-conditioned archive manifest:"
cat "${archive_manifest_file}"

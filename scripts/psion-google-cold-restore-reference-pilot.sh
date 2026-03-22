#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
POLICY_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_checkpoint_archive_policy_v1.json"
ARCHIVE_MANIFEST_URI=""
MANIFEST_OUT=""

usage() {
  cat <<'EOF'
Usage: psion-google-cold-restore-reference-pilot.sh [options] <archive_manifest_uri>

Options:
  --manifest-out <path>      Write the generated cold-restore manifest to one local path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${ARCHIVE_MANIFEST_URI}" ]]; then
        ARCHIVE_MANIFEST_URI="$1"
        shift
      else
        echo "error: unexpected extra argument $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${ARCHIVE_MANIFEST_URI}" ]]; then
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

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(compute_sha256 "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "error: sha256 mismatch for ${path}: expected ${expected}, found ${actual}" >&2
    exit 1
  fi
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

archive_manifest_file="${tmpdir}/archive_manifest.json"
gcloud storage cp --quiet "${ARCHIVE_MANIFEST_URI}" "${archive_manifest_file}" >/dev/null

repo_git_revision="$(jq -r '.repo_git_revision' "${archive_manifest_file}")"
current_revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ "${repo_git_revision}" != "${current_revision}" ]]; then
  echo "error: archive manifest revision ${repo_git_revision} does not match current checkout ${current_revision}" >&2
  exit 1
fi

checkpoint_prefix="$(jq -r '.checkpoint_prefix' "${archive_manifest_file}")"
cold_restore_prefix_name="$(jq -r '.cold_restore_prefix' "${POLICY_FILE}")"
cold_restore_manifest_name="$(jq -r '.cold_restore_manifest_name' "${POLICY_FILE}")"
run_id="$(jq -r '.run_id' "${archive_manifest_file}")"
checkpoint_ref="$(jq -r '.checkpoint_ref' "${archive_manifest_file}")"
recovery_mode="$(jq -r '.recovery_mode' "${archive_manifest_file}")"

download_dir="${tmpdir}/checkpoint_dir"
resume_output_dir="${tmpdir}/resume_probe"
mkdir -p "${download_dir}" "${resume_output_dir}"

while IFS= read -r artifact_json; do
  artifact_kind="$(jq -r '.artifact_kind' <<<"${artifact_json}")"
  remote_uri="$(jq -r '.remote_uri' <<<"${artifact_json}")"
  expected_sha256="$(jq -r '.sha256' <<<"${artifact_json}")"
  dest_name="$(basename "${remote_uri}")"
  dest_path="${download_dir}/${dest_name}"
  gcloud storage cp --quiet "${remote_uri}" "${dest_path}" >/dev/null
  verify_sha256 "${dest_path}" "${expected_sha256}"
done < <(jq -c '.artifacts[]' "${archive_manifest_file}")

(
  cd "${REPO_ROOT}"
  cargo run -p psionic-train --example psion_reference_pilot_resume_probe -- "${download_dir}" "${resume_output_dir}" >/dev/null
)

resume_probe_file="${resume_output_dir}/psion_reference_pilot_resume_probe.json"
resume_probe_sha256="$(compute_sha256 "${resume_probe_file}")"
probe_recovery_mode="$(jq -r '.recovery_mode' "${resume_probe_file}")"
if [[ "${probe_recovery_mode}" != "${recovery_mode}" ]]; then
  echo "error: resume probe recovery mode ${probe_recovery_mode} does not match archive manifest ${recovery_mode}" >&2
  exit 1
fi

restore_id="psion-google-cold-restore-$(date -u '+%Y%m%dt%H%M%Sz' | tr '[:upper:]' '[:lower:]')"
resume_probe_uri="${checkpoint_prefix}/${cold_restore_prefix_name}/${restore_id}/psion_reference_pilot_resume_probe.json"
cold_restore_manifest_uri="${checkpoint_prefix}/${cold_restore_prefix_name}/${restore_id}/${cold_restore_manifest_name}"
created_at_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
cold_restore_manifest_file="${tmpdir}/${cold_restore_manifest_name}"

jq -n \
  --arg schema_version "psion.google_reference_checkpoint_cold_restore_manifest.v1" \
  --arg restore_id "${restore_id}" \
  --arg created_at_utc "${created_at_utc}" \
  --arg source_archive_manifest_uri "${ARCHIVE_MANIFEST_URI}" \
  --arg run_id "${run_id}" \
  --arg checkpoint_ref "${checkpoint_ref}" \
  --arg recovery_mode "${recovery_mode}" \
  --arg resume_probe_uri "${resume_probe_uri}" \
  --arg cold_restore_manifest_uri "${cold_restore_manifest_uri}" \
  --arg resume_probe_sha256 "${resume_probe_sha256}" \
  '{
    schema_version: $schema_version,
    restore_id: $restore_id,
    created_at_utc: $created_at_utc,
    source_archive_manifest_uri: $source_archive_manifest_uri,
    run_id: $run_id,
    checkpoint_ref: $checkpoint_ref,
    recovery_mode: $recovery_mode,
    result: "bounded_success",
    cold_restore_manifest_uri: $cold_restore_manifest_uri,
    resume_probe_uri: $resume_probe_uri,
    resume_probe_sha256: $resume_probe_sha256,
    detail: "Cold restore downloaded the archived checkpoint bundle from GCS, verified object digests, and replayed resume_from_last_stable_checkpoint through the reference pilot resume probe."
  }' > "${cold_restore_manifest_file}"

gcloud storage cp --quiet "${resume_probe_file}" "${resume_probe_uri}" >/dev/null
gcloud storage cp --quiet "${cold_restore_manifest_file}" "${cold_restore_manifest_uri}" >/dev/null
wait_for_object "${resume_probe_uri}"
wait_for_object "${cold_restore_manifest_uri}"

if [[ -n "${MANIFEST_OUT}" ]]; then
  cp "${cold_restore_manifest_file}" "${MANIFEST_OUT}"
fi

echo "cold restore manifest:"
cat "${cold_restore_manifest_file}"

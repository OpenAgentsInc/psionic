#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
policy_file="${repo_root}/fixtures/parameter_golf/google/parameter_golf_google_input_package_policy_v1.json"
bringup_report="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_single_h100_bringup.json"
parameter_golf_root="${HOME}/code/parameter-golf"
output_dir="${repo_root}/fixtures/parameter_golf/google"
created_at_utc="2026-03-23T00:00:00Z"
upload=false
upload_report=""

usage() {
  cat <<'EOF' >&2
Usage: parameter-golf-google-package-inputs.sh [options]

Options:
  --policy <path>               Override the committed input-package policy.
  --bringup-report <path>       Override the committed single-H100 bring-up report.
  --parameter-golf-root <path>  Local parameter-golf clone used to pin the downloader revision.
  --output-dir <path>           Directory that receives the generated contract bundle.
  --created-at-utc <timestamp>  Override the retained creation timestamp.
  --upload                      Upload the descriptor and archive to the configured Google bucket.
  --upload-report <path>        Write a machine-readable upload receipt after remote verification.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      policy_file="$2"
      shift 2
      ;;
    --bringup-report)
      bringup_report="$2"
      shift 2
      ;;
    --parameter-golf-root)
      parameter_golf_root="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --created-at-utc)
      created_at_utc="$2"
      shift 2
      ;;
    --upload)
      upload=true
      shift
      ;;
    --upload-report)
      upload_report="$2"
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

sha256_of_string() {
  local value="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "${value}" | sha256sum | awk '{print $1}'
  else
    printf '%s' "${value}" | shasum -a 256 | awk '{print $1}'
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

if [[ ! -d "${parameter_golf_root}/.git" ]]; then
  echo "error: parameter-golf repo not found at ${parameter_golf_root}" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

package_id="$(jq -r '.package_id' "${policy_file}")"
descriptor_uri="$(jq -r '.descriptor_uri' "${policy_file}")"
archive_uri="$(jq -r '.archive_uri' "${policy_file}")"
materialization_mode="$(jq -r '.materialization.mode' "${policy_file}")"
max_local_working_set_gb="$(jq -r '.materialization.max_local_working_set_gb' "${policy_file}")"
psionic_git_revision="$(git rev-parse HEAD)"
parameter_golf_git_revision="$(git -C "${parameter_golf_root}" rev-parse HEAD)"
dataset_ref="$(jq -r '.dataset_key.dataset_ref' "${bringup_report}")"
dataset_version="$(jq -r '.dataset_key.version' "${bringup_report}")"
dataset_manifest_sha256="$(jq -r '.dataset_manifest_digest' "${bringup_report}")"
tokenizer_family="$(jq -r '.tokenizer_digest.family' "${bringup_report}")"
tokenizer_sha256="$(jq -r '.tokenizer_digest.tokenizer_digest' "${bringup_report}")"
vocab_size="$(jq -r '.tokenizer_digest.vocab_size' "${bringup_report}")"
train_shards="$(jq -r '.train_shard_count' "${bringup_report}")"
validation_shards="$(jq -r '.validation_shard_count' "${bringup_report}")"
validation_identity="$(jq -r '.validation_identity' "${bringup_report}")"
validation_identity_sha256="$(sha256_of_string "${validation_identity}")"
variant="$(jq -r '.variant' "${bringup_report}")"

contract_file="${tmpdir}/parameter_golf_google_input_contract_v1.json"
manifest_file="${tmpdir}/parameter_golf_google_input_package_manifest_v1.json"
descriptor_file="${tmpdir}/parameter_golf_google_input_package_descriptor_v1.json"
archive_root="${tmpdir}/archive_root"
archive_path="${tmpdir}/parameter_golf_google_input_package_v1.tar.gz"

jq -n \
  --arg schema_version "parameter_golf.google_input_contract.v1" \
  --arg package_id "${package_id}" \
  --arg created_at_utc "${created_at_utc}" \
  --arg psionic_git_revision "${psionic_git_revision}" \
  --arg clone_url "https://github.com/openai/parameter-golf.git" \
  --arg parameter_golf_git_revision "${parameter_golf_git_revision}" \
  --arg repo_id "$(jq -r '.matched_fineweb.repo_id' "${policy_file}")" \
  --arg remote_root_prefix "$(jq -r '.matched_fineweb.remote_root_prefix' "${policy_file}")" \
  --arg dataset_ref "${dataset_ref}" \
  --arg dataset_version "${dataset_version}" \
  --arg dataset_manifest_sha256 "${dataset_manifest_sha256}" \
  --arg variant "${variant}" \
  --argjson train_shards "${train_shards}" \
  --argjson validation_shards "${validation_shards}" \
  --arg validation_identity "${validation_identity}" \
  --arg dataset_local_relative_root "$(jq -r '.dataset.local_relative_root' "${policy_file}")" \
  --arg tokenizer_family "${tokenizer_family}" \
  --arg tokenizer_sha256 "${tokenizer_sha256}" \
  --argjson vocab_size "${vocab_size}" \
  --arg tokenizer_local_relative_path "$(jq -r '.tokenizer.local_relative_path' "${policy_file}")" \
  --arg tokenizer_vocab_relative_path "$(jq -r '.tokenizer.vocab_relative_path' "${policy_file}")" \
  --arg challenge_cache_command "python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80" \
  --argjson consumers "$(jq '.consumers' "${policy_file}")" \
  '{
    schema_version: $schema_version,
    package_id: $package_id,
    created_at_utc: $created_at_utc,
    psionic_git_revision: $psionic_git_revision,
    parameter_golf_repo: {
      clone_url: $clone_url,
      git_revision: $parameter_golf_git_revision
    },
    matched_fineweb: {
      repo_id: $repo_id,
      remote_root_prefix: $remote_root_prefix
    },
    dataset: {
      dataset_ref: $dataset_ref,
      version: $dataset_version,
      dataset_manifest_sha256: $dataset_manifest_sha256,
      variant: $variant,
      train_shards: $train_shards,
      validation_shards: $validation_shards,
      validation_identity: $validation_identity,
      local_relative_root: $dataset_local_relative_root
    },
    tokenizer: {
      family: $tokenizer_family,
      sha256: $tokenizer_sha256,
      vocab_size: $vocab_size,
      local_relative_path: $tokenizer_local_relative_path,
      vocab_relative_path: $tokenizer_vocab_relative_path
    },
    public_cache_bootstrap: {
      downloader_entrypoint: "data/cached_challenge_fineweb.py",
      challenge_cache_command: $challenge_cache_command
    },
    consumers: $consumers
  }' > "${contract_file}"

contract_sha256="$(compute_sha256 "${contract_file}")"
bringup_report_sha256="$(compute_sha256 "${bringup_report}")"

jq -n \
  --arg schema_version "psion.google_training_input_package_manifest.v1" \
  --arg package_id "${package_id}" \
  --arg created_at_utc "${created_at_utc}" \
  --arg git_revision "${psionic_git_revision}" \
  --arg repo_clone_url "https://github.com/OpenAgentsInc/psionic.git" \
  --arg materialization_mode "${materialization_mode}" \
  --argjson max_local_working_set_gb "${max_local_working_set_gb}" \
  --arg dataset_manifest_sha256 "${dataset_manifest_sha256}" \
  --arg tokenizer_sha256 "${tokenizer_sha256}" \
  --arg validation_identity_sha256 "${validation_identity_sha256}" \
  --arg input_contract_sha256 "${contract_sha256}" \
  --arg bringup_report_sha256 "${bringup_report_sha256}" \
  --arg dataset_ref "${dataset_ref}" \
  --arg dataset_version "${dataset_version}" \
  --arg validation_identity "${validation_identity}" \
  --arg parameter_golf_git_revision "${parameter_golf_git_revision}" \
  --argjson train_shards "${train_shards}" \
  --argjson validation_shards "${validation_shards}" \
  --argjson benchmark_execution_posture "$(jq '.benchmark_execution_posture' "${policy_file}")" \
  '{
    schema_version: $schema_version,
    package_id: $package_id,
    created_at_utc: $created_at_utc,
    git_revision: $git_revision,
    repo_clone_url: $repo_clone_url,
    materialization: {
      mode: $materialization_mode,
      max_local_working_set_gb: $max_local_working_set_gb
    },
    benchmark_execution_posture: $benchmark_execution_posture,
    key_digests: {
      dataset_manifest_sha256: $dataset_manifest_sha256,
      tokenizer_sha256: $tokenizer_sha256,
      validation_identity_sha256: $validation_identity_sha256,
      input_contract_sha256: $input_contract_sha256,
      parameter_golf_single_h100_bringup_report_sha256: $bringup_report_sha256,
      dataset_ref: $dataset_ref,
      dataset_version: $dataset_version,
      validation_identity: $validation_identity,
      parameter_golf_git_revision: $parameter_golf_git_revision,
      train_shards: $train_shards,
      validation_shards: $validation_shards
    },
    artifacts: [
      {
        artifact_id: "parameter_golf_google_input_contract_v1",
        kind: "parameter_golf_input_contract",
        package_path: "parameter_golf_google_input_contract_v1.json",
        overlay_target: null,
        sha256: $input_contract_sha256
      }
    ]
  }' > "${manifest_file}"

mkdir -p "${archive_root}"
cp "${contract_file}" "${archive_root}/parameter_golf_google_input_contract_v1.json"
cp "${manifest_file}" "${archive_root}/psion_google_reference_input_package_manifest.json"
(
  cd "${archive_root}"
  tar \
    --sort=name \
    --mtime="${created_at_utc}" \
    --owner=0 \
    --group=0 \
    --numeric-owner \
    -czf "${archive_path}" .
)

archive_sha256="$(compute_sha256 "${archive_path}")"
manifest_sha256="$(compute_sha256 "${manifest_file}")"

jq -n \
  --arg schema_version "psion.google_training_input_package_descriptor.v1" \
  --arg package_id "${package_id}" \
  --arg created_at_utc "${created_at_utc}" \
  --arg git_revision "${psionic_git_revision}" \
  --arg descriptor_uri "${descriptor_uri}" \
  --arg archive_uri "${archive_uri}" \
  --arg archive_sha256 "${archive_sha256}" \
  --arg manifest_path_in_archive "psion_google_reference_input_package_manifest.json" \
  --arg manifest_sha256 "${manifest_sha256}" \
  --arg materialization_mode "${materialization_mode}" \
  --argjson max_local_working_set_gb "${max_local_working_set_gb}" \
  --argjson benchmark_execution_posture "$(jq '.benchmark_execution_posture' "${manifest_file}")" \
  --argjson key_digests "$(jq '.key_digests' "${manifest_file}")" \
  '{
    schema_version: $schema_version,
    package_id: $package_id,
    created_at_utc: $created_at_utc,
    git_revision: $git_revision,
    descriptor_uri: $descriptor_uri,
    archive_uri: $archive_uri,
    archive_sha256: $archive_sha256,
    manifest_path_in_archive: $manifest_path_in_archive,
    manifest_sha256: $manifest_sha256,
    materialization: {
      mode: $materialization_mode,
      max_local_working_set_gb: $max_local_working_set_gb
    },
    benchmark_execution_posture: $benchmark_execution_posture,
    key_digests: $key_digests
  }' > "${descriptor_file}"

mkdir -p "${output_dir}"
cp "${contract_file}" "${output_dir}/parameter_golf_google_input_contract_v1.json"
cp "${manifest_file}" "${output_dir}/parameter_golf_google_input_package_manifest_v1.json"
cp "${descriptor_file}" "${output_dir}/parameter_golf_google_input_package_descriptor_v1.json"
cp "${archive_path}" "${output_dir}/parameter_golf_google_input_package_v1.tar.gz"

if [[ "${upload}" == "true" ]]; then
  gcloud storage cp --quiet \
    "${descriptor_file}" \
    "${descriptor_uri}" >/dev/null
  gcloud storage cp --quiet \
    "${archive_path}" \
    "${archive_uri}" >/dev/null
  wait_for_object "${descriptor_uri}"
  wait_for_object "${archive_uri}"
fi

if [[ -n "${upload_report}" ]]; then
  wait_for_object "${descriptor_uri}"
  wait_for_object "${archive_uri}"
  descriptor_remote_json="$(gcloud storage objects describe "${descriptor_uri}" --format=json)"
  archive_remote_json="$(gcloud storage objects describe "${archive_uri}" --format=json)"
  descriptor_local_path="${output_dir}/parameter_golf_google_input_package_descriptor_v1.json"
  archive_local_path="${output_dir}/parameter_golf_google_input_package_v1.tar.gz"
  descriptor_local_sha256="$(compute_sha256 "${descriptor_local_path}")"
  archive_local_sha256="$(compute_sha256 "${archive_local_path}")"
  descriptor_local_size="$(wc -c < "${descriptor_local_path}" | tr -d ' ')"
  archive_local_size="$(wc -c < "${archive_local_path}" | tr -d ' ')"
  mkdir -p "$(dirname "${upload_report}")"
  jq -n \
    --arg schema_version "parameter_golf.google_input_package_upload.v1" \
    --arg checked_at_utc "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    --arg package_id "${package_id}" \
    --arg descriptor_local_path "${descriptor_local_path}" \
    --arg descriptor_local_sha256 "${descriptor_local_sha256}" \
    --arg descriptor_local_size "${descriptor_local_size}" \
    --arg archive_local_path "${archive_local_path}" \
    --arg archive_local_sha256 "${archive_local_sha256}" \
    --arg archive_local_size "${archive_local_size}" \
    --arg uploaded "${upload}" \
    --argjson descriptor_remote "${descriptor_remote_json}" \
    --argjson archive_remote "${archive_remote_json}" \
    '{
      schema_version: $schema_version,
      checked_at_utc: $checked_at_utc,
      package_id: $package_id,
      uploaded_in_this_invocation: ($uploaded == "true"),
      visibility_verified: true,
      detail: "The committed Parameter Golf immutable input-package descriptor and archive were verified in the real openagentsgemini bucket and tied back to the local committed artifacts by path and SHA-256.",
      descriptor: {
        local_path: $descriptor_local_path,
        local_sha256: $descriptor_local_sha256,
        local_size_bytes: ($descriptor_local_size | tonumber),
        remote_object: $descriptor_remote
      },
      archive: {
        local_path: $archive_local_path,
        local_sha256: $archive_local_sha256,
        local_size_bytes: ($archive_local_size | tonumber),
        remote_object: $archive_remote
      }
    }' > "${upload_report}"
fi

jq -n \
  --arg package_id "${package_id}" \
  --arg descriptor_path "${output_dir}/parameter_golf_google_input_package_descriptor_v1.json" \
  --arg archive_path "${output_dir}/parameter_golf_google_input_package_v1.tar.gz" \
  --arg uploaded "${upload}" \
  --arg upload_report_path "${upload_report}" \
  '{
    package_id: $package_id,
    descriptor_path: $descriptor_path,
    archive_path: $archive_path,
    uploaded: ($uploaded == "true"),
    upload_report_path: (if $upload_report_path == "" then null else $upload_report_path end)
  }'

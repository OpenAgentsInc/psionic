#!/usr/bin/env bash

set -Eeuo pipefail

LOG_FILE="/var/log/psion-google-startup.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
export HOME="${HOME:-/root}"

RUN_PHASE="bootstrap"
FAILURE_CODE="bounded_success"
FAILURE_DETAIL=""
TRAINING_EXIT_CODE=0
BOOTSTRAP_STARTED_AT_UTC=""
BOOTSTRAP_FINISHED_AT_UTC=""
TRAINING_STARTED_AT_UTC=""
TRAINING_FINISHED_AT_UTC=""
CHECKPOINT_COMPLETED_AT_UTC=""
TEARDOWN_STARTED_AT_UTC=""
ARCHIVE_MANIFEST_URI=""
COLD_RESTORE_MANIFEST_URI=""
FINALIZED=0
GPU_MONITOR_PID=""
LAUNCH_MANIFEST_URI=""
LAUNCH_MANIFEST_FILE=""
RUN_ROOT=""
REPO_DIR=""
EVENT_LOG_FILE=""
GPU_SAMPLE_INTERVAL_SECONDS=5

timestamp_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

metadata_value() {
  local key="$1"
  curl -fsSL -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}"
}

metadata_value_optional() {
  local key="$1"
  curl -fsSL -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}" \
    2>/dev/null || true
}

compute_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

gcs_download() {
  local gs_uri="$1"
  local dest_path="$2"
  local bucket object encoded_object token
  bucket="${gs_uri#gs://}"
  bucket="${bucket%%/*}"
  object="${gs_uri#gs://${bucket}/}"
  encoded_object="$(python3 - "${object}" <<'PY'
import sys
import urllib.parse
print(urllib.parse.quote(sys.argv[1], safe=""))
PY
)"
  token="$(
    curl -fsSL -H "Metadata-Flavor: Google" \
      http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token | \
      jq -r '.access_token'
  )"
  mkdir -p "$(dirname "${dest_path}")"
  curl -fsSL -H "Authorization: Bearer ${token}" \
    "https://storage.googleapis.com/storage/v1/b/${bucket}/o/${encoded_object}?alt=media" \
    -o "${dest_path}"
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

ensure_rust_toolchain() {
  local toolchain="$1"
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain "${toolchain}"
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"
}

check_free_disk_gb() {
  local path="$1"
  local low_watermark_gb="$2"
  local available_gb
  available_gb="$(df --output=avail -BG "${path}" | tail -1 | tr -dc '0-9')"
  if [[ -z "${available_gb}" ]]; then
    echo "error: failed to determine free disk space for ${path}" >&2
    exit 1
  fi
  if (( available_gb < low_watermark_gb )); then
    RUN_PHASE="cost_guardrail"
    echo "error: free disk ${available_gb}GB is below low watermark ${low_watermark_gb}GB" >&2
    exit 1
  fi
}

emit_event() {
  local event_kind="$1"
  local detail="$2"
  if [[ -z "${EVENT_LOG_FILE}" ]]; then
    return 0
  fi
  jq -nc \
    --arg timestamp_utc "$(timestamp_utc)" \
    --arg phase "${RUN_PHASE}" \
    --arg event_kind "${event_kind}" \
    --arg detail "${detail}" \
    --arg failure_code "${FAILURE_CODE}" \
    '{
      timestamp_utc: $timestamp_utc,
      phase: $phase,
      event_kind: $event_kind,
      detail: $detail,
      failure_code: $failure_code
    }' >> "${EVENT_LOG_FILE}"
}

phase_failure_code() {
  case "$1" in
    driver_check)
      printf '%s' "driver_runtime_failure"
      ;;
    training)
      printf '%s' "training_divergence"
      ;;
    checkpoint_restore)
      printf '%s' "checkpoint_restore_failure"
      ;;
    artifact_upload|finalizing)
      printf '%s' "artifact_upload_failure"
      ;;
    cost_guardrail)
      printf '%s' "cost_guardrail_abort"
      ;;
    *)
      printf '%s' "bootstrap_failure"
      ;;
  esac
}

first_matching_file() {
  local search_root="$1"
  local pattern="$2"
  find "${search_root}" -type f -name "${pattern}" | sort | head -n 1 || true
}

start_gpu_monitor() {
  local sample_file="$1"
  local sample_interval_seconds="${2:-5}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  printf 'timestamp,name,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,temperature_gpu_c,power_draw_w\n' > "${sample_file}"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits >> "${sample_file}"
      sleep "${sample_interval_seconds}"
    done
  ) &
  GPU_MONITOR_PID="$!"
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
}

validate_accelerated_training_evidence() {
  local launch_manifest_file="$1"
  local output_dir="$2"
  local gpu_sample_file="$3"
  local policy_file="$4"
  local expected_backend trainer_lane_id stage_receipt_file observed_backend accelerator_backed
  local sample_interval_seconds warmup_seconds minimum_post_warmup_samples warmup_samples
  local require_nonzero_gpu_utilization require_nonzero_gpu_memory_residency validation_json
  local considered_samples nonzero_utilization_samples nonzero_memory_samples avg_gpu_util_percent
  local max_gpu_util_percent max_memory_used_mib

  if [[ ! -f "${launch_manifest_file}" ]]; then
    FAILURE_CODE="training_runtime_failure"
    FAILURE_DETAIL="accelerator validation could not read ${launch_manifest_file}"
    return 1
  fi

  expected_backend="$(
    jq -r '.expected_execution_backend // .training.expected_execution_backend // "cpu"' \
      "${launch_manifest_file}"
  )"
  trainer_lane_id="$(
    jq -r '.trainer_lane_id // .training.trainer_lane_id // "unknown_trainer_lane"' \
      "${launch_manifest_file}"
  )"
  if [[ "${expected_backend}" != "cuda" ]]; then
    return 0
  fi

  stage_receipt_file="$(first_matching_file "${output_dir}" '*stage_receipt.json')"
  if [[ -z "${stage_receipt_file}" ]]; then
    FAILURE_CODE="training_runtime_failure"
    FAILURE_DETAIL="accelerator validation could not find a stage receipt under ${output_dir}"
    return 1
  fi

  observed_backend="$(jq -r '.delivered_execution.runtime_backend // empty' "${stage_receipt_file}")"
  accelerator_backed="$(jq -r '.accelerator_execution.accelerator_backed // false' "${stage_receipt_file}")"
  if [[ "${observed_backend}" != "cuda" ]]; then
    FAILURE_CODE="training_runtime_failure"
    FAILURE_DETAIL="accelerator validation expected backend cuda for ${trainer_lane_id}, found ${observed_backend:-unset}"
    return 1
  fi
  if [[ "${accelerator_backed}" != "true" ]]; then
    FAILURE_CODE="training_runtime_failure"
    FAILURE_DETAIL="accelerator validation expected accelerator_execution.accelerator_backed=true for ${trainer_lane_id}"
    return 1
  fi

  if [[ ! -f "${gpu_sample_file}" || ! -s "${gpu_sample_file}" ]]; then
    FAILURE_CODE="training_runtime_failure"
    FAILURE_DETAIL="accelerator validation could not read GPU samples from ${gpu_sample_file}"
    return 1
  fi

  sample_interval_seconds="$(jq -r '.gpu_sample_interval_seconds // 5' "${policy_file}")"
  warmup_seconds="$(jq -r '.accelerator_validation.warmup_seconds // 0' "${policy_file}")"
  minimum_post_warmup_samples="$(
    jq -r '.accelerator_validation.minimum_post_warmup_samples // 1' "${policy_file}"
  )"
  require_nonzero_gpu_utilization="$(
    jq -r '.accelerator_validation.require_nonzero_gpu_utilization // true' "${policy_file}"
  )"
  require_nonzero_gpu_memory_residency="$(
    jq -r '.accelerator_validation.require_nonzero_gpu_memory_residency // true' "${policy_file}"
  )"
  if [[ ! "${sample_interval_seconds}" =~ ^[0-9]+$ ]] || (( sample_interval_seconds < 1 )); then
    sample_interval_seconds=5
  fi
  if [[ ! "${warmup_seconds}" =~ ^[0-9]+$ ]] || (( warmup_seconds < 0 )); then
    warmup_seconds=0
  fi
  if [[ ! "${minimum_post_warmup_samples}" =~ ^[0-9]+$ ]] || (( minimum_post_warmup_samples < 1 )); then
    minimum_post_warmup_samples=1
  fi
  warmup_samples=$(( (warmup_seconds + sample_interval_seconds - 1) / sample_interval_seconds ))

  validation_json="$(
    awk -F',' -v warmup_samples="${warmup_samples}" '
      BEGIN {
        raw_samples = 0
        considered = 0
        nonzero_util = 0
        nonzero_mem = 0
        sum_gpu = 0
        max_gpu = 0
        max_mem = 0
      }
      NR == 1 {
        next
      }
      NF < 6 {
        next
      }
      {
        raw_samples += 1
        if (raw_samples <= warmup_samples) {
          next
        }
        gpu_util = $3
        mem_used = $5
        gsub(/^[ \t]+|[ \t]+$/, "", gpu_util)
        gsub(/^[ \t]+|[ \t]+$/, "", mem_used)
        gpu_util += 0
        mem_used += 0
        considered += 1
        sum_gpu += gpu_util
        if (gpu_util > 0) {
          nonzero_util += 1
        }
        if (mem_used > 0) {
          nonzero_mem += 1
        }
        if (gpu_util > max_gpu) {
          max_gpu = gpu_util
        }
        if (mem_used > max_mem) {
          max_mem = mem_used
        }
      }
      END {
        avg_gpu = considered > 0 ? sum_gpu / considered : 0
        printf("{\"considered_samples\":%d,\"nonzero_utilization_samples\":%d,\"nonzero_memory_samples\":%d,\"avg_gpu_util_percent\":%.2f,\"max_gpu_util_percent\":%.2f,\"max_memory_used_mib\":%.2f}", considered, nonzero_util, nonzero_mem, avg_gpu, max_gpu, max_mem)
      }
    ' "${gpu_sample_file}"
  )"

  considered_samples="$(jq -r '.considered_samples' <<<"${validation_json}")"
  nonzero_utilization_samples="$(jq -r '.nonzero_utilization_samples' <<<"${validation_json}")"
  nonzero_memory_samples="$(jq -r '.nonzero_memory_samples' <<<"${validation_json}")"
  avg_gpu_util_percent="$(jq -r '.avg_gpu_util_percent' <<<"${validation_json}")"
  max_gpu_util_percent="$(jq -r '.max_gpu_util_percent' <<<"${validation_json}")"
  max_memory_used_mib="$(jq -r '.max_memory_used_mib' <<<"${validation_json}")"

  if (( considered_samples < minimum_post_warmup_samples )); then
    FAILURE_CODE="training_runtime_failure"
    FAILURE_DETAIL="accelerator validation observed ${considered_samples} post-warmup GPU samples for ${trainer_lane_id}, required ${minimum_post_warmup_samples}"
    return 1
  fi
  if [[ "${require_nonzero_gpu_utilization}" == "true" ]] && (( nonzero_utilization_samples == 0 )); then
    FAILURE_CODE="training_runtime_failure"
    FAILURE_DETAIL="accelerator validation observed zero post-warmup GPU utilization for ${trainer_lane_id}"
    return 1
  fi
  if [[ "${require_nonzero_gpu_memory_residency}" == "true" ]] && (( nonzero_memory_samples == 0 )); then
    FAILURE_CODE="training_runtime_failure"
    FAILURE_DETAIL="accelerator validation observed zero post-warmup GPU memory residency for ${trainer_lane_id}"
    return 1
  fi

  emit_event \
    "accelerator_validation_passed" \
    "Validated ${trainer_lane_id} on cuda with ${considered_samples} post-warmup samples, avg GPU util ${avg_gpu_util_percent}%, peak GPU util ${max_gpu_util_percent}%, and peak memory ${max_memory_used_mib} MiB."
  return 0
}

finalize_run() {
  local finalize_status tmpdir final_manifest_file failure_log_file failure_log_uri final_manifest_uri
  if (( FINALIZED == 1 )); then
    return 0
  fi
  FINALIZED=1
  trap - ERR TERM INT
  set +e

  RUN_PHASE="finalizing"
  TEARDOWN_STARTED_AT_UTC="${TEARDOWN_STARTED_AT_UTC:-$(timestamp_utc)}"
  stop_gpu_monitor
  emit_event "finalization_started" "Collecting host facts and uploading retained run evidence."

  if [[ -n "${REPO_DIR}" && -x "${REPO_DIR}/scripts/psion-google-finalize-run.sh" ]]; then
    bash "${REPO_DIR}/scripts/psion-google-finalize-run.sh" \
      --run-root "${RUN_ROOT}" \
      --repo-dir "${REPO_DIR}" \
      --run-id "${PSION_RUN_ID}" \
      --launch-manifest-uri "${LAUNCH_MANIFEST_URI}" \
      --failure-code "${FAILURE_CODE}" \
      --failure-detail "${FAILURE_DETAIL}" \
      --training-exit-code "${TRAINING_EXIT_CODE}" \
      --bootstrap-started-at-utc "${BOOTSTRAP_STARTED_AT_UTC}" \
      --bootstrap-finished-at-utc "${BOOTSTRAP_FINISHED_AT_UTC}" \
      --training-started-at-utc "${TRAINING_STARTED_AT_UTC}" \
      --training-finished-at-utc "${TRAINING_FINISHED_AT_UTC}" \
      --checkpoint-completed-at-utc "${CHECKPOINT_COMPLETED_AT_UTC}" \
      --teardown-started-at-utc "${TEARDOWN_STARTED_AT_UTC}" \
      --archive-manifest-uri "${ARCHIVE_MANIFEST_URI}" \
      --cold-restore-manifest-uri "${COLD_RESTORE_MANIFEST_URI}"
    finalize_status=$?
  else
    tmpdir="$(mktemp -d)"
    failure_log_file="${tmpdir}/psion_google_startup_failure.log"
    final_manifest_file="${tmpdir}/psion_google_run_final_manifest.json"
    failure_log_uri="${PSION_BUCKET_URL}/runs/${PSION_RUN_ID}/final/psion_google_startup_failure.log"
    final_manifest_uri="${PSION_BUCKET_URL}/runs/${PSION_RUN_ID}/final/psion_google_run_final_manifest.json"
    cp "${LOG_FILE}" "${failure_log_file}"
    gcloud storage cp --quiet "${failure_log_file}" "${failure_log_uri}" >/dev/null
    jq -n \
      --arg schema_version "psion.google_run_final_manifest.v1" \
      --arg created_at_utc "$(timestamp_utc)" \
      --arg run_id "${PSION_RUN_ID}" \
      --arg result_classification "${FAILURE_CODE}" \
      --arg failure_detail "${FAILURE_DETAIL}" \
      --arg launch_manifest_uri "${LAUNCH_MANIFEST_URI}" \
      --arg failure_log_uri "${failure_log_uri}" \
      --arg bootstrap_started_at_utc "${BOOTSTRAP_STARTED_AT_UTC}" \
      --arg bootstrap_finished_at_utc "${BOOTSTRAP_FINISHED_AT_UTC}" \
      --arg training_started_at_utc "${TRAINING_STARTED_AT_UTC}" \
      --arg training_finished_at_utc "${TRAINING_FINISHED_AT_UTC}" \
      --arg checkpoint_completed_at_utc "${CHECKPOINT_COMPLETED_AT_UTC}" \
      --arg teardown_started_at_utc "${TEARDOWN_STARTED_AT_UTC}" \
      '{
        schema_version: $schema_version,
        created_at_utc: $created_at_utc,
        run_id: $run_id,
        result_classification: $result_classification,
        failure_detail: (if $failure_detail == "" then null else $failure_detail end),
        launch_manifest_uri: $launch_manifest_uri,
        bootstrap_timeline_utc: {
          bootstrap_started_at_utc: (if $bootstrap_started_at_utc == "" then null else $bootstrap_started_at_utc end),
          bootstrap_finished_at_utc: (if $bootstrap_finished_at_utc == "" then null else $bootstrap_finished_at_utc end),
          training_started_at_utc: (if $training_started_at_utc == "" then null else $training_started_at_utc end),
          training_finished_at_utc: (if $training_finished_at_utc == "" then null else $training_finished_at_utc end),
          checkpoint_completed_at_utc: (if $checkpoint_completed_at_utc == "" then null else $checkpoint_completed_at_utc end),
          teardown_started_at_utc: (if $teardown_started_at_utc == "" then null else $teardown_started_at_utc end)
        },
        retained_objects: [
          {
            artifact_kind: "startup_failure_log",
            evidence_role: "log",
            remote_uri: $failure_log_uri
          }
        ],
        detail: "Startup failed before the repo-owned host finalizer became available, so the fallback manifest preserves the typed failure code, timing, and startup log."
      }' > "${final_manifest_file}"
    gcloud storage cp --quiet "${final_manifest_file}" "${final_manifest_uri}" >/dev/null
    rm -rf "${tmpdir}"
    finalize_status=0
  fi

  if (( finalize_status != 0 )); then
    echo "error: run finalization failed with status ${finalize_status}" >&2
    TRAINING_EXIT_CODE="${finalize_status}"
    if [[ "${FAILURE_CODE}" == "bounded_success" ]]; then
      FAILURE_CODE="artifact_upload_failure"
      FAILURE_DETAIL="run finalization failed with status ${finalize_status}"
    fi
    return "${finalize_status}"
  fi

  emit_event "finalization_completed" "Uploaded final manifest and retained artifacts."
  return 0
}

on_error() {
  local exit_code="$?"
  local line_no="$1"
  if (( FINALIZED == 1 )); then
    exit "${exit_code}"
  fi
  TRAINING_EXIT_CODE="${exit_code}"
  if [[ "${FAILURE_CODE}" == "bounded_success" ]]; then
    FAILURE_CODE="$(phase_failure_code "${RUN_PHASE}")"
  fi
  if [[ -z "${FAILURE_DETAIL}" ]]; then
    FAILURE_DETAIL="phase=${RUN_PHASE} line=${line_no} exit=${exit_code}"
  fi
  echo "error: ${FAILURE_DETAIL}" >&2
  emit_event "run_error" "${FAILURE_DETAIL}"
  finalize_run || true
  exit "${exit_code}"
}

on_signal() {
  local signal_name="$1"
  TRAINING_EXIT_CODE=143
  FAILURE_CODE="operator_abort"
  FAILURE_DETAIL="received signal ${signal_name}"
  emit_event "signal_received" "${FAILURE_DETAIL}"
  finalize_run || true
  exit 143
}

trap 'on_error $LINENO' ERR
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

main() {
  local run_id bucket_url repo_clone_url git_revision pre_training_command training_command workspace_root
  local output_subdir log_subdir scratch_subdir repo_subdir low_disk_watermark_gb toolchain
  local apt_packages
  local input_package_descriptor_uri input_package_archive_uri input_package_archive_sha256
  local input_package_manifest_sha256 input_materialization_mode
  local post_training_archive_command post_training_restore_command
  local output_dir log_dir scratch_dir repo_dir entrypoint_file input_root
  local input_descriptor_path input_archive_path input_package_root input_manifest_path
  local archive_manifest_file cold_restore_manifest_file archive_exit restore_exit

  BOOTSTRAP_STARTED_AT_UTC="$(timestamp_utc)"

  run_id="$(metadata_value psion-run-id)"
  bucket_url="$(metadata_value psion-bucket-url)"
  repo_clone_url="$(metadata_value psion-repo-clone-url)"
  git_revision="$(metadata_value psion-git-revision)"
  pre_training_command="$(metadata_value_optional psion-pre-training-command)"
  training_command="$(metadata_value psion-training-command)"
  workspace_root="$(metadata_value psion-workspace-root)"
  output_subdir="$(metadata_value psion-output-subdir)"
  log_subdir="$(metadata_value psion-log-subdir)"
  scratch_subdir="$(metadata_value psion-scratch-subdir)"
  repo_subdir="$(metadata_value psion-repo-subdir)"
  low_disk_watermark_gb="$(metadata_value psion-low-disk-watermark-gb)"
  toolchain="$(metadata_value psion-rust-toolchain)"
  apt_packages="$(metadata_value psion-apt-packages)"
  input_package_descriptor_uri="$(metadata_value psion-input-package-descriptor-uri)"
  input_package_archive_uri="$(metadata_value psion-input-package-archive-uri)"
  input_package_archive_sha256="$(metadata_value psion-input-package-archive-sha256)"
  input_package_manifest_sha256="$(metadata_value psion-input-package-manifest-sha256)"
  input_materialization_mode="$(metadata_value psion-input-materialization-mode)"
  post_training_archive_command="$(metadata_value_optional psion-post-training-archive-command)"
  post_training_restore_command="$(metadata_value_optional psion-post-training-restore-command)"
  LAUNCH_MANIFEST_URI="$(metadata_value psion-launch-manifest-uri)"

  echo "psion google startup: run=${run_id} repo=${repo_clone_url} revision=${git_revision}"
  echo "psion google startup: bucket=${bucket_url}"

  export PSION_RUN_ID="${run_id}"
  export PSION_BUCKET_URL="${bucket_url}"
  export PSIONIC_REMOTE_TRAINING_PROVIDER="google_cloud"
  export PSIONIC_REMOTE_TRAINING_PROFILE_ID="$(
    jq -r '.profile_id // "unknown_profile"' "${LAUNCH_MANIFEST_FILE}"
  )"
  export PSIONIC_REMOTE_TRAINING_LANE_ID="$(
    jq -r '.trainer_lane_id // .training.trainer_lane_id // "unknown_trainer_lane"' "${LAUNCH_MANIFEST_FILE}"
  )"
  export PSIONIC_REMOTE_TRAINING_REPO_REVISION="workspace@${git_revision}"

  RUN_ROOT="${workspace_root}/runs/${run_id}"
  output_dir="${RUN_ROOT}/${output_subdir}"
  log_dir="${RUN_ROOT}/${log_subdir}"
  scratch_dir="${RUN_ROOT}/${scratch_subdir}"
  repo_dir="${RUN_ROOT}/${repo_subdir}"
  input_root="${RUN_ROOT}/inputs"
  EVENT_LOG_FILE="${log_dir}/psion_google_run_events.jsonl"
  mkdir -p "${output_dir}" "${log_dir}" "${scratch_dir}" "${input_root}"
  LAUNCH_MANIFEST_FILE="${scratch_dir}/psion_google_launch_manifest.json"
  gcs_download "${LAUNCH_MANIFEST_URI}" "${LAUNCH_MANIFEST_FILE}"
  emit_event "bootstrap_started" "Startup script began bounded Google single-node training setup."

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  # shellcheck disable=SC2086
  apt-get install -y ${apt_packages}
  emit_event "apt_packages_installed" "Required operator and build packages installed."

  RUN_PHASE="driver_check"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not found on the selected GPU image" >&2
    exit 1
  fi
  nvidia-smi
  emit_event "driver_verified" "nvidia-smi was present and the GPU runtime responded."

  RUN_PHASE="bootstrap"
  ensure_rust_toolchain "${toolchain}"
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"
  emit_event "rust_ready" "Rust toolchain available for repo-owned pilot commands."

  check_free_disk_gb "${RUN_ROOT}" "${low_disk_watermark_gb}"

  git clone "${repo_clone_url}" "${repo_dir}"
  git -C "${repo_dir}" fetch --depth=1 origin "${git_revision}"
  git -C "${repo_dir}" checkout --detach "${git_revision}"
  git config --global --add safe.directory "${repo_dir}"
  REPO_DIR="${repo_dir}"
  GPU_SAMPLE_INTERVAL_SECONDS="$(
    jq -r '.gpu_sample_interval_seconds // 5' \
      "${repo_dir}/fixtures/psion/google/psion_google_host_observability_policy_v1.json"
  )"
  if [[ ! "${GPU_SAMPLE_INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] || (( GPU_SAMPLE_INTERVAL_SECONDS < 1 )); then
    GPU_SAMPLE_INTERVAL_SECONDS=5
  fi
  emit_event "repo_checked_out" "Repo cloned and detached to the requested revision."

  RUN_PHASE="bootstrap"
  input_descriptor_path="${input_root}/psion_google_input_package_descriptor.json"
  input_archive_path="${input_root}/psion_google_input_package.tar.gz"
  input_package_root="${input_root}/package"
  input_manifest_path="${input_package_root}/psion_google_reference_input_package_manifest.json"
  mkdir -p "${input_package_root}"
  gcs_download "${input_package_descriptor_uri}" "${input_descriptor_path}"
  gcs_download "${input_package_archive_uri}" "${input_archive_path}"
  verify_sha256 "${input_archive_path}" "${input_package_archive_sha256}"
  tar -xzf "${input_archive_path}" -C "${input_package_root}"
  verify_sha256 "${input_manifest_path}" "${input_package_manifest_sha256}"

  while IFS= read -r artifact_json; do
    package_path="$(jq -r '.package_path' <<<"${artifact_json}")"
    expected_sha256="$(jq -r '.sha256' <<<"${artifact_json}")"
    verify_sha256 "${input_package_root}/${package_path}" "${expected_sha256}"
  done < <(jq -c '.artifacts[]' "${input_manifest_path}")

  if [[ "${input_materialization_mode}" == "stage_to_local_disk_overlay_repo" ]]; then
    cp -a "${input_package_root}/repo_overlay/." "${repo_dir}/"
    git config --global --add safe.directory "${repo_dir}"
  fi
  emit_event "inputs_materialized" "Immutable input package was downloaded, verified, and overlaid."

  check_free_disk_gb "${RUN_ROOT}" "${low_disk_watermark_gb}"

  export PSION_WORKSPACE_ROOT="${RUN_ROOT}"
  export PSION_OUTPUT_DIR="${output_dir}"
  export PSION_LOG_DIR="${log_dir}"
  export PSION_SCRATCH_DIR="${scratch_dir}"
  export PSION_REPO_DIR="${repo_dir}"
  export PSION_REPO_GIT_REVISION="${git_revision}"
  export PSION_INPUT_PACKAGE_ROOT="${input_package_root}"
  export PSION_INPUT_PACKAGE_DESCRIPTOR="${input_descriptor_path}"
  export PSION_INPUT_PACKAGE_DESCRIPTOR_URI="${input_package_descriptor_uri}"
  export PSION_INPUT_PACKAGE_MANIFEST="${input_manifest_path}"
  export PSION_INPUT_MATERIALIZATION_MODE="${input_materialization_mode}"
  export CARGO_TARGET_DIR="${scratch_dir}/cargo-target"

  entrypoint_file="${RUN_ROOT}/psion_google_training_entrypoint.sh"
  if [[ -n "${pre_training_command}" ]]; then
    local pre_training_entrypoint_file
    pre_training_entrypoint_file="${RUN_ROOT}/psion_google_pre_training_entrypoint.sh"
    cat > "${pre_training_entrypoint_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.cargo/env"
export PSION_RUN_ID="${PSION_RUN_ID}"
export PSION_BUCKET_URL="${PSION_BUCKET_URL}"
export PSION_WORKSPACE_ROOT="${PSION_WORKSPACE_ROOT}"
export PSION_OUTPUT_DIR="${PSION_OUTPUT_DIR}"
export PSION_LOG_DIR="${PSION_LOG_DIR}"
export PSION_SCRATCH_DIR="${PSION_SCRATCH_DIR}"
export PSION_REPO_DIR="${PSION_REPO_DIR}"
export PSION_REPO_GIT_REVISION="${PSION_REPO_GIT_REVISION}"
export PSION_INPUT_PACKAGE_ROOT="${PSION_INPUT_PACKAGE_ROOT}"
export PSION_INPUT_PACKAGE_DESCRIPTOR="${PSION_INPUT_PACKAGE_DESCRIPTOR}"
export PSION_INPUT_PACKAGE_DESCRIPTOR_URI="${PSION_INPUT_PACKAGE_DESCRIPTOR_URI}"
export PSION_INPUT_PACKAGE_MANIFEST="${PSION_INPUT_PACKAGE_MANIFEST}"
export PSION_INPUT_MATERIALIZATION_MODE="${PSION_INPUT_MATERIALIZATION_MODE}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR}"
cd "${repo_dir}"
${pre_training_command}
EOF
    chmod +x "${pre_training_entrypoint_file}"
    emit_event "pre_training_started" "Running the repo-owned pre-training command before measured accelerator validation."
    set +e
    "${pre_training_entrypoint_file}" \
      > "${log_dir}/pre_training.stdout.log" \
      2> "${log_dir}/pre_training.stderr.log"
    TRAINING_EXIT_CODE="$?"
    set -e
    if (( TRAINING_EXIT_CODE != 0 )); then
      FAILURE_CODE="bootstrap_failure"
      FAILURE_DETAIL="pre-training command exited with status ${TRAINING_EXIT_CODE}"
      emit_event "pre_training_failed" "${FAILURE_DETAIL}"
      finalize_run
      exit "${TRAINING_EXIT_CODE}"
    fi
    emit_event "pre_training_completed" "The repo-owned pre-training command completed before the measured training window."
    check_free_disk_gb "${RUN_ROOT}" "${low_disk_watermark_gb}"
  fi

  cat > "${entrypoint_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.cargo/env"
export PSION_RUN_ID="${PSION_RUN_ID}"
export PSION_BUCKET_URL="${PSION_BUCKET_URL}"
export PSION_WORKSPACE_ROOT="${PSION_WORKSPACE_ROOT}"
export PSION_OUTPUT_DIR="${PSION_OUTPUT_DIR}"
export PSION_LOG_DIR="${PSION_LOG_DIR}"
export PSION_SCRATCH_DIR="${PSION_SCRATCH_DIR}"
export PSION_REPO_DIR="${PSION_REPO_DIR}"
export PSION_REPO_GIT_REVISION="${PSION_REPO_GIT_REVISION}"
export PSION_INPUT_PACKAGE_ROOT="${PSION_INPUT_PACKAGE_ROOT}"
export PSION_INPUT_PACKAGE_DESCRIPTOR="${PSION_INPUT_PACKAGE_DESCRIPTOR}"
export PSION_INPUT_PACKAGE_DESCRIPTOR_URI="${PSION_INPUT_PACKAGE_DESCRIPTOR_URI}"
export PSION_INPUT_PACKAGE_MANIFEST="${PSION_INPUT_PACKAGE_MANIFEST}"
export PSION_INPUT_MATERIALIZATION_MODE="${PSION_INPUT_MATERIALIZATION_MODE}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR}"
cd "${repo_dir}"
${training_command}
EOF
  chmod +x "${entrypoint_file}"

  BOOTSTRAP_FINISHED_AT_UTC="$(timestamp_utc)"
  RUN_PHASE="training"
  TRAINING_STARTED_AT_UTC="$(timestamp_utc)"
  emit_event "training_started" "Launching the repo-owned training command on the GPU host."
  start_gpu_monitor "${log_dir}/psion_google_gpu_samples.csv" "${GPU_SAMPLE_INTERVAL_SECONDS}"
  set +e
  "${entrypoint_file}" \
    > "${log_dir}/training.stdout.log" \
    2> "${log_dir}/training.stderr.log"
  TRAINING_EXIT_CODE="$?"
  set -e
  TRAINING_FINISHED_AT_UTC="$(timestamp_utc)"

  if (( TRAINING_EXIT_CODE != 0 )); then
    FAILURE_CODE="training_divergence"
    FAILURE_DETAIL="training command exited with status ${TRAINING_EXIT_CODE}"
    emit_event "training_failed" "${FAILURE_DETAIL}"
    finalize_run
    exit "${TRAINING_EXIT_CODE}"
  fi
  stop_gpu_monitor
  emit_event "training_completed" "The repo-owned training command completed without a non-zero process exit."
  if ! validate_accelerated_training_evidence \
    "${LAUNCH_MANIFEST_FILE}" \
    "${output_dir}" \
    "${log_dir}/psion_google_gpu_samples.csv" \
    "${repo_dir}/fixtures/psion/google/psion_google_host_observability_policy_v1.json"; then
    emit_event "accelerator_validation_failed" "${FAILURE_DETAIL}"
    finalize_run
    exit 1
  fi

  if [[ -n "${post_training_archive_command}" ]]; then
    RUN_PHASE="artifact_upload"
    archive_manifest_file="${scratch_dir}/psion_google_checkpoint_archive_manifest.json"
    export PSION_ARCHIVE_MANIFEST_OUT="${archive_manifest_file}"
    emit_event "checkpoint_archive_started" "Running the lane-specific post-training archive command."
    set +e
    bash -lc "${post_training_archive_command}" \
      > "${log_dir}/checkpoint_archive.stdout.log" \
      2> "${log_dir}/checkpoint_archive.stderr.log"
    archive_exit="$?"
    set -e
    if (( archive_exit != 0 )); then
      FAILURE_CODE="artifact_upload_failure"
      FAILURE_DETAIL="archive command exited with status ${archive_exit}"
      TRAINING_EXIT_CODE="${archive_exit}"
      emit_event "checkpoint_archive_failed" "${FAILURE_DETAIL}"
      finalize_run
      exit "${archive_exit}"
    fi
    if [[ ! -f "${archive_manifest_file}" ]]; then
      FAILURE_CODE="artifact_upload_failure"
      FAILURE_DETAIL="archive command did not write ${archive_manifest_file}"
      emit_event "checkpoint_archive_failed" "${FAILURE_DETAIL}"
      finalize_run
      exit 1
    fi
    ARCHIVE_MANIFEST_URI="$(jq -r '.archive_manifest_uri // empty' "${archive_manifest_file}")"
    if [[ -z "${ARCHIVE_MANIFEST_URI}" ]]; then
      FAILURE_CODE="artifact_upload_failure"
      FAILURE_DETAIL="archive manifest did not include archive_manifest_uri"
      emit_event "checkpoint_archive_failed" "${FAILURE_DETAIL}"
      finalize_run
      exit 1
    fi
    CHECKPOINT_COMPLETED_AT_UTC="$(timestamp_utc)"
    export PSION_ARCHIVE_MANIFEST_URI="${ARCHIVE_MANIFEST_URI}"
    emit_event "checkpoint_archived" "Checkpoint archive manifest uploaded to GCS."
  fi

  if [[ -n "${post_training_restore_command}" && -n "${ARCHIVE_MANIFEST_URI}" ]]; then
    RUN_PHASE="checkpoint_restore"
    cold_restore_manifest_file="${scratch_dir}/psion_google_cold_restore_manifest.json"
    export PSION_COLD_RESTORE_MANIFEST_OUT="${cold_restore_manifest_file}"
    emit_event "checkpoint_restore_started" "Running the lane-specific cold-restore command."
    set +e
    bash -lc "${post_training_restore_command}" \
      > "${log_dir}/checkpoint_restore.stdout.log" \
      2> "${log_dir}/checkpoint_restore.stderr.log"
    restore_exit="$?"
    set -e
    if (( restore_exit != 0 )); then
      FAILURE_CODE="checkpoint_restore_failure"
      FAILURE_DETAIL="restore command exited with status ${restore_exit}"
      TRAINING_EXIT_CODE="${restore_exit}"
      emit_event "checkpoint_restore_failed" "${FAILURE_DETAIL}"
      finalize_run
      exit "${restore_exit}"
    fi
    if [[ ! -f "${cold_restore_manifest_file}" ]]; then
      FAILURE_CODE="checkpoint_restore_failure"
      FAILURE_DETAIL="restore command did not write ${cold_restore_manifest_file}"
      emit_event "checkpoint_restore_failed" "${FAILURE_DETAIL}"
      finalize_run
      exit 1
    fi
    COLD_RESTORE_MANIFEST_URI="$(jq -r '.cold_restore_manifest_uri // empty' "${cold_restore_manifest_file}")"
    if [[ -z "${COLD_RESTORE_MANIFEST_URI}" ]]; then
      FAILURE_CODE="checkpoint_restore_failure"
      FAILURE_DETAIL="cold restore manifest did not include cold_restore_manifest_uri"
      emit_event "checkpoint_restore_failed" "${FAILURE_DETAIL}"
      finalize_run
      exit 1
    fi
    emit_event "checkpoint_restored" "Cold restore probe replayed the lane-specific archive posture."
  fi

  finalize_run
  echo "psion google startup: run completed with failure_code=${FAILURE_CODE}"
}

main "$@"

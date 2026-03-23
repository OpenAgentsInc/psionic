#!/usr/bin/env bash

set -euo pipefail

RUN_ROOT="${RUN_ROOT:-}"
REPO_DIR="${REPO_DIR:-}"
RUN_ID="${RUN_ID:-}"
LAUNCH_MANIFEST_URI="${LAUNCH_MANIFEST_URI:-}"
FAILURE_CODE="${FAILURE_CODE:-bounded_success}"
FAILURE_DETAIL="${FAILURE_DETAIL:-}"
TRAINING_EXIT_CODE="${TRAINING_EXIT_CODE:-0}"
BOOTSTRAP_STARTED_AT_UTC="${BOOTSTRAP_STARTED_AT_UTC:-}"
BOOTSTRAP_FINISHED_AT_UTC="${BOOTSTRAP_FINISHED_AT_UTC:-}"
TRAINING_STARTED_AT_UTC="${TRAINING_STARTED_AT_UTC:-}"
TRAINING_FINISHED_AT_UTC="${TRAINING_FINISHED_AT_UTC:-}"
CHECKPOINT_COMPLETED_AT_UTC="${CHECKPOINT_COMPLETED_AT_UTC:-}"
TEARDOWN_STARTED_AT_UTC="${TEARDOWN_STARTED_AT_UTC:-}"
ARCHIVE_MANIFEST_URI="${ARCHIVE_MANIFEST_URI:-}"
COLD_RESTORE_MANIFEST_URI="${COLD_RESTORE_MANIFEST_URI:-}"

usage() {
  cat <<'EOF'
Usage: psion-google-finalize-run.sh [options]

Options:
  --run-root <path>                    Local run root on the training host.
  --repo-dir <path>                    Checked-out repo root used for the run.
  --run-id <id>                        Declared run id.
  --launch-manifest-uri <uri>          Launch manifest object for the run.
  --failure-code <code>                Typed run result classification.
  --failure-detail <detail>            Human-readable failure detail.
  --training-exit-code <code>          Training command exit code.
  --bootstrap-started-at-utc <ts>      UTC bootstrap start timestamp.
  --bootstrap-finished-at-utc <ts>     UTC bootstrap finish timestamp.
  --training-started-at-utc <ts>       UTC training start timestamp.
  --training-finished-at-utc <ts>      UTC training finish timestamp.
  --checkpoint-completed-at-utc <ts>   UTC checkpoint completion timestamp.
  --teardown-started-at-utc <ts>       UTC teardown start timestamp.
  --archive-manifest-uri <uri>         Uploaded checkpoint archive manifest URI.
  --cold-restore-manifest-uri <uri>    Uploaded cold-restore manifest URI.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --launch-manifest-uri)
      LAUNCH_MANIFEST_URI="$2"
      shift 2
      ;;
    --failure-code)
      FAILURE_CODE="$2"
      shift 2
      ;;
    --failure-detail)
      FAILURE_DETAIL="$2"
      shift 2
      ;;
    --training-exit-code)
      TRAINING_EXIT_CODE="$2"
      shift 2
      ;;
    --bootstrap-started-at-utc)
      BOOTSTRAP_STARTED_AT_UTC="$2"
      shift 2
      ;;
    --bootstrap-finished-at-utc)
      BOOTSTRAP_FINISHED_AT_UTC="$2"
      shift 2
      ;;
    --training-started-at-utc)
      TRAINING_STARTED_AT_UTC="$2"
      shift 2
      ;;
    --training-finished-at-utc)
      TRAINING_FINISHED_AT_UTC="$2"
      shift 2
      ;;
    --checkpoint-completed-at-utc)
      CHECKPOINT_COMPLETED_AT_UTC="$2"
      shift 2
      ;;
    --teardown-started-at-utc)
      TEARDOWN_STARTED_AT_UTC="$2"
      shift 2
      ;;
    --archive-manifest-uri)
      ARCHIVE_MANIFEST_URI="$2"
      shift 2
      ;;
    --cold-restore-manifest-uri)
      COLD_RESTORE_MANIFEST_URI="$2"
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

if [[ -z "${RUN_ROOT}" || -z "${REPO_DIR}" || -z "${RUN_ID}" || -z "${LAUNCH_MANIFEST_URI}" ]]; then
  echo "error: --run-root, --repo-dir, --run-id, and --launch-manifest-uri are required" >&2
  exit 1
fi

POLICY_FILE="${REPO_DIR}/fixtures/psion/google/psion_google_host_observability_policy_v1.json"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

if [[ ! -f "${POLICY_FILE}" ]]; then
  echo "error: observability policy not found: ${POLICY_FILE}" >&2
  exit 1
fi

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
  for attempt in 1 2 3 4 5; do
    if gcloud storage ls "${object_path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: object ${object_path} did not become visible" >&2
  exit 1
}

metadata_value_optional() {
  local path="$1"
  curl -fsSL --max-time 1 -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/${path}" 2>/dev/null || true
}

safe_command_output() {
  local output=""
  if "$@" >/tmp/psion-google-command.out 2>/tmp/psion-google-command.err; then
    output="$(cat /tmp/psion-google-command.out)"
  else
    output="$(cat /tmp/psion-google-command.err)"
  fi
  printf '%s' "${output}"
}

classify_local_role() {
  local file_name="$1"
  case "${file_name}" in
    *checkpoint_manifest.json) printf '%s' "manifest" ;;
    *optimizer_state.json) printf '%s' "manifest" ;;
    *stage_receipt.json|*observability_receipt.json|*route_class_evaluation_receipt.json|*refusal_calibration_receipt.json) printf '%s' "receipt" ;;
    *pilot_pretraining_bundle.json|*benchmark_eval.json|*summary.json) printf '%s' "manifest" ;;
    *.json) printf '%s' "manifest" ;;
    *.jsonl) printf '%s' "event_log" ;;
    *.log) printf '%s' "log" ;;
    *.csv) printf '%s' "metrics" ;;
    *.safetensors) printf '%s' "artifact" ;;
    *) printf '%s' "artifact" ;;
  esac
}

first_matching_file() {
  local search_root="$1"
  local pattern="$2"
  find "${search_root}" -type f -name "${pattern}" | sort | head -n 1 || true
}

bucket_url="$(jq -r '.bucket_url' "${POLICY_FILE}")"
host_prefix_name="$(jq -r '.artifact_paths.host_prefix' "${POLICY_FILE}")"
logs_prefix_name="$(jq -r '.artifact_paths.logs_prefix' "${POLICY_FILE}")"
receipts_prefix_name="$(jq -r '.artifact_paths.receipts_prefix' "${POLICY_FILE}")"
artifacts_prefix_name="$(jq -r '.artifact_paths.artifacts_prefix' "${POLICY_FILE}")"
final_prefix_name="$(jq -r '.artifact_paths.final_prefix' "${POLICY_FILE}")"
event_log_name="$(jq -r '.event_log_name' "${POLICY_FILE}")"
timeline_name="$(jq -r '.timeline_name' "${POLICY_FILE}")"
host_facts_name="$(jq -r '.host_facts_name' "${POLICY_FILE}")"
runtime_snapshot_name="$(jq -r '.runtime_snapshot_name' "${POLICY_FILE}")"
gpu_samples_name="$(jq -r '.gpu_samples_name' "${POLICY_FILE}")"
gpu_summary_name="$(jq -r '.gpu_summary_name' "${POLICY_FILE}")"
accelerator_validation_receipt_name="$(
  jq -r '.accelerator_validation_receipt_name' "${POLICY_FILE}"
)"
outcome_name="$(jq -r '.outcome_name' "${POLICY_FILE}")"
manifest_of_manifests_name="$(jq -r '.manifest_of_manifests_name' "${POLICY_FILE}")"
final_manifest_name="$(jq -r '.final_manifest_name' "${POLICY_FILE}")"

run_prefix="${bucket_url}/runs/${RUN_ID}"
host_prefix="${run_prefix}/${host_prefix_name}"
logs_prefix="${run_prefix}/${logs_prefix_name}"
receipts_prefix="${run_prefix}/${receipts_prefix_name}"
artifacts_prefix="${run_prefix}/${artifacts_prefix_name}"
final_prefix="${run_prefix}/${final_prefix_name}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}" /tmp/psion-google-command.out /tmp/psion-google-command.err' EXIT

artifact_records_file="${tmpdir}/artifact_records.jsonl"
launch_manifest_file="${tmpdir}/launch_manifest.json"
gcloud storage cp --quiet "${LAUNCH_MANIFEST_URI}" "${launch_manifest_file}" >/dev/null

project_id="$(jq -r '.project_id' "${launch_manifest_file}")"
zone="$(jq -r '.zone' "${launch_manifest_file}")"
instance_name="$(jq -r '.instance_name' "${launch_manifest_file}")"
profile_id="$(jq -r '.profile_id' "${launch_manifest_file}")"
profile_label="$(jq -r '.profile_label' "${launch_manifest_file}")"
machine_type="$(jq -r '.machine.machine_type' "${launch_manifest_file}")"
accelerator_type="$(jq -r '.machine.accelerator_type' "${launch_manifest_file}")"
accelerator_count="$(jq -r '.machine.accelerator_count' "${launch_manifest_file}")"
network_name="$(jq -r '.network.network' "${launch_manifest_file}")"
subnetwork_name="$(jq -r '.network.subnetwork' "${launch_manifest_file}")"
external_ip_enabled="$(jq -r '.network.external_ip' "${launch_manifest_file}")"
boot_disk_type="$(jq -r '.storage.boot_disk_type' "${launch_manifest_file}")"
boot_disk_gb="$(jq -r '.storage.boot_disk_gb' "${launch_manifest_file}")"
low_disk_watermark_gb="$(jq -r '.storage.low_disk_watermark_gb' "${launch_manifest_file}")"
launch_created_at_utc="$(jq -r '.created_at_utc' "${launch_manifest_file}")"
startup_script_uri="$(jq -r '.artifact_paths.startup_script_uri' "${launch_manifest_file}")"
preflight_uri="$(jq -r '.artifact_paths.quota_preflight_uri' "${launch_manifest_file}")"
input_descriptor_uri="$(jq -r '.input_package.descriptor_uri' "${launch_manifest_file}")"
input_archive_uri="$(jq -r '.input_package.archive_uri' "${launch_manifest_file}")"
input_archive_sha256="$(jq -r '.input_package.archive_sha256' "${launch_manifest_file}")"
input_manifest_sha256="$(jq -r '.input_package.manifest_sha256' "${launch_manifest_file}")"
trainer_lane_id="$(jq -r '.trainer_lane_id // .training.trainer_lane_id // "unknown_trainer_lane"' "${launch_manifest_file}")"
expected_execution_backend="$(
  jq -r '.expected_execution_backend // .training.expected_execution_backend // "cpu"' \
    "${launch_manifest_file}"
)"
image_project="$(jq -r '.image.image_project' "${launch_manifest_file}")"
image_family="$(jq -r '.image.image_family' "${launch_manifest_file}")"
image_name="$(jq -r '.image.image_name' "${launch_manifest_file}")"
image_self_link="$(jq -r '.image.image_self_link' "${launch_manifest_file}")"
image_id="$(jq -r '.image.image_id' "${launch_manifest_file}")"
image_creation_timestamp="$(jq -r '.image.image_creation_timestamp' "${launch_manifest_file}")"

output_dir="${RUN_ROOT}/output"
log_dir="${RUN_ROOT}/logs"
scratch_dir="${RUN_ROOT}/scratch"
event_log_path="${log_dir}/${event_log_name}"
timeline_path="${scratch_dir}/${timeline_name}"
host_facts_path="${scratch_dir}/${host_facts_name}"
runtime_snapshot_path="${scratch_dir}/${runtime_snapshot_name}"
gpu_samples_path="${log_dir}/${gpu_samples_name}"
gpu_summary_path="${scratch_dir}/${gpu_summary_name}"
accelerator_validation_path="${scratch_dir}/${accelerator_validation_receipt_name}"
outcome_path="${scratch_dir}/${outcome_name}"

mkdir -p "${log_dir}" "${scratch_dir}"

collect_remote_sha_record() {
  local artifact_kind="$1"
  local evidence_role="$2"
  local remote_uri="$3"
  local local_copy="$4"
  gcloud storage cp --quiet "${remote_uri}" "${local_copy}" >/dev/null
  jq -nc \
    --arg artifact_kind "${artifact_kind}" \
    --arg evidence_role "${evidence_role}" \
    --arg remote_uri "${remote_uri}" \
    --arg sha256 "$(compute_sha256 "${local_copy}")" \
    --arg byte_length "$(wc -c < "${local_copy}" | tr -d ' ')" \
    '{
      artifact_kind: $artifact_kind,
      evidence_role: $evidence_role,
      remote_uri: $remote_uri,
      sha256: $sha256,
      byte_length: ($byte_length | tonumber),
      source_mode: "remote_reference"
    }' >> "${artifact_records_file}"
}

upload_local_artifact() {
  local artifact_kind="$1"
  local evidence_role="$2"
  local local_path="$3"
  local remote_uri="$4"
  local sha256
  sha256="$(compute_sha256 "${local_path}")"
  gcloud storage cp --quiet "${local_path}" "${remote_uri}" >/dev/null
  wait_for_object "${remote_uri}"
  jq -nc \
    --arg artifact_kind "${artifact_kind}" \
    --arg evidence_role "${evidence_role}" \
    --arg local_path "${local_path}" \
    --arg remote_uri "${remote_uri}" \
    --arg sha256 "${sha256}" \
    --arg byte_length "$(wc -c < "${local_path}" | tr -d ' ')" \
    '{
      artifact_kind: $artifact_kind,
      evidence_role: $evidence_role,
      local_path: $local_path,
      remote_uri: $remote_uri,
      sha256: $sha256,
      byte_length: ($byte_length | tonumber),
      source_mode: "local_upload"
    }' >> "${artifact_records_file}"
}

record_reference_object() {
  local artifact_kind="$1"
  local evidence_role="$2"
  local remote_uri="$3"
  local sha256="$4"
  local byte_length="$5"
  jq -nc \
    --arg artifact_kind "${artifact_kind}" \
    --arg evidence_role "${evidence_role}" \
    --arg remote_uri "${remote_uri}" \
    --arg sha256 "${sha256}" \
    --arg byte_length "${byte_length}" \
    '{
      artifact_kind: $artifact_kind,
      evidence_role: $evidence_role,
      remote_uri: $remote_uri,
      sha256: $sha256,
      byte_length: (if $byte_length == "" then null else ($byte_length | tonumber) end),
      source_mode: "referenced_object"
    }' >> "${artifact_records_file}"
}

instance_zone_metadata="$(metadata_value_optional 'instance/zone')"
machine_type_metadata="$(metadata_value_optional 'instance/machine-type')"
instance_id="$(metadata_value_optional 'instance/id')"
internal_ip="$(metadata_value_optional 'instance/network-interfaces/0/ip')"
network_interface_name="$(metadata_value_optional 'instance/network-interfaces/0/name')"
network_interface_mac="$(metadata_value_optional 'instance/network-interfaces/0/mac')"
hostname_value="$(metadata_value_optional 'instance/hostname')"

driver_version=""
cuda_version=""
gpu_name=""
nvidia_smi_output=""
if command -v nvidia-smi >/dev/null 2>&1; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '\r' || true)"
  cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \\([^ ]*\\).*/\\1/p' | head -n 1 || true)"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\r' || true)"
  nvidia_smi_output="$(safe_command_output nvidia-smi)"
fi

cpu_summary="$(safe_command_output lscpu)"
memory_summary="$(safe_command_output free -b)"
disk_summary="$(safe_command_output df -B1)"
uptime_summary="$(safe_command_output uptime)"
kernel_summary="$(safe_command_output uname -a)"

jq -n \
  --arg schema_version "psion.google_host_facts.v1" \
  --arg collected_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --arg project_id "${project_id}" \
  --arg zone "${zone}" \
  --arg instance_name "${instance_name}" \
  --arg instance_id "${instance_id}" \
  --arg hostname "${hostname_value}" \
  --arg profile_id "${profile_id}" \
  --arg profile_label "${profile_label}" \
  --arg machine_type "${machine_type}" \
  --arg accelerator_type "${accelerator_type}" \
  --argjson accelerator_count "${accelerator_count}" \
  --arg gpu_name "${gpu_name}" \
  --arg network "${network_name}" \
  --arg subnetwork "${subnetwork_name}" \
  --arg internal_ip "${internal_ip}" \
  --arg network_interface_name "${network_interface_name}" \
  --arg network_interface_mac "${network_interface_mac}" \
  --arg external_ip_enabled "${external_ip_enabled}" \
  --arg boot_disk_type "${boot_disk_type}" \
  --argjson boot_disk_gb "${boot_disk_gb}" \
  --argjson low_disk_watermark_gb "${low_disk_watermark_gb}" \
  --arg image_project "${image_project}" \
  --arg image_family "${image_family}" \
  --arg image_name "${image_name}" \
  --arg image_self_link "${image_self_link}" \
  --arg image_id "${image_id}" \
  --arg image_creation_timestamp "${image_creation_timestamp}" \
  --arg metadata_zone "${instance_zone_metadata}" \
  --arg metadata_machine_type "${machine_type_metadata}" \
  --arg driver_version "${driver_version}" \
  --arg cuda_version "${cuda_version}" \
  '{
    schema_version: $schema_version,
    collected_at_utc: $collected_at_utc,
    run_id: $run_id,
    project_id: $project_id,
    zone: $zone,
    instance: {
      name: $instance_name,
      id: $instance_id,
      hostname: $hostname,
      metadata_zone: $metadata_zone,
      metadata_machine_type: $metadata_machine_type
    },
    profile: {
      profile_id: $profile_id,
      profile_label: $profile_label
    },
    machine: {
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: $accelerator_count,
      observed_gpu_name: (if $gpu_name == "" then null else $gpu_name end)
    },
    network: {
      network: $network,
      subnetwork: $subnetwork,
      internal_ip: (if $internal_ip == "" then null else $internal_ip end),
      interface_name: (if $network_interface_name == "" then null else $network_interface_name end),
      interface_mac: (if $network_interface_mac == "" then null else $network_interface_mac end),
      external_ip_enabled: ($external_ip_enabled == "true")
    },
    storage: {
      boot_disk_type: $boot_disk_type,
      boot_disk_gb: $boot_disk_gb,
      low_disk_watermark_gb: $low_disk_watermark_gb,
      local_disk_authority: false
    },
    image: {
      image_project: $image_project,
      image_family: $image_family,
      image_name: $image_name,
      image_self_link: $image_self_link,
      image_id: $image_id,
      image_creation_timestamp: $image_creation_timestamp
    },
    runtime: {
      driver_version: (if $driver_version == "" then null else $driver_version end),
      cuda_version: (if $cuda_version == "" then null else $cuda_version end)
    }
  }' > "${host_facts_path}"

jq -n \
  --arg schema_version "psion.google_runtime_snapshot.v1" \
  --arg captured_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --arg nvidia_smi "${nvidia_smi_output}" \
  --arg cpu_summary "${cpu_summary}" \
  --arg memory_summary "${memory_summary}" \
  --arg disk_summary "${disk_summary}" \
  --arg uptime_summary "${uptime_summary}" \
  --arg kernel_summary "${kernel_summary}" \
  '{
    schema_version: $schema_version,
    captured_at_utc: $captured_at_utc,
    run_id: $run_id,
    nvidia_smi: (if $nvidia_smi == "" then null else $nvidia_smi end),
    cpu_summary: $cpu_summary,
    memory_summary: $memory_summary,
    disk_summary: $disk_summary,
    uptime_summary: $uptime_summary,
    kernel_summary: $kernel_summary
  }' > "${runtime_snapshot_path}"

if [[ -f "${gpu_samples_path}" && -s "${gpu_samples_path}" ]]; then
  gpu_stats_json="$(
    awk -F',' '
      BEGIN {
        count = 0
        sum_gpu = 0
        sum_mem = 0
        max_gpu = 0
        max_mem = 0
      }
      {
        if (NR == 1) {
          next
        }
        if (NF < 6) {
          next
        }
        gpu_util = $3
        mem_util = $4
        mem_used = $5
        mem_total = $6
        gsub(/^[ \t]+|[ \t]+$/, "", gpu_util)
        gsub(/^[ \t]+|[ \t]+$/, "", mem_util)
        gsub(/^[ \t]+|[ \t]+$/, "", mem_used)
        gsub(/^[ \t]+|[ \t]+$/, "", mem_total)
        sum_gpu += gpu_util + 0
        sum_mem += mem_util + 0
        if ((gpu_util + 0) > max_gpu) {
          max_gpu = gpu_util + 0
        }
        if ((mem_used + 0) > max_mem) {
          max_mem = mem_used + 0
        }
        final_mem_total = mem_total + 0
        count += 1
      }
      END {
        if (count == 0) {
          printf("{\"sample_count\":0}")
        } else {
          printf("{\"sample_count\":%d,\"avg_gpu_util_percent\":%.2f,\"max_gpu_util_percent\":%.2f,\"avg_memory_util_percent\":%.2f,\"max_memory_used_mib\":%.2f,\"observed_memory_total_mib\":%.2f}", count, sum_gpu / count, max_gpu, sum_mem / count, max_mem, final_mem_total)
        }
      }' "${gpu_samples_path}"
  )"
else
  gpu_stats_json='{"sample_count":0}'
fi

jq -n \
  --arg schema_version "psion.google_gpu_summary.v1" \
  --arg captured_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --argjson stats "${gpu_stats_json}" \
  '{
    schema_version: $schema_version,
    captured_at_utc: $captured_at_utc,
    run_id: $run_id,
    stats: $stats
  }' > "${gpu_summary_path}"

final_result_classification="${FAILURE_CODE}"
final_failure_detail="${FAILURE_DETAIL}"
stage_receipt_file="$(first_matching_file "${output_dir}" '*stage_receipt.json')"
observability_receipt_file="$(first_matching_file "${output_dir}" '*observability_receipt.json')"
sample_interval_seconds="$(jq -r '.gpu_sample_interval_seconds // 5' "${POLICY_FILE}")"
warmup_seconds="$(jq -r '.accelerator_validation.warmup_seconds // 0' "${POLICY_FILE}")"
minimum_post_warmup_samples="$(
  jq -r '.accelerator_validation.minimum_post_warmup_samples // 1' "${POLICY_FILE}"
)"
require_nonzero_gpu_utilization="$(
  jq -r '.accelerator_validation.require_nonzero_gpu_utilization // true' "${POLICY_FILE}"
)"
require_nonzero_gpu_memory_residency="$(
  jq -r '.accelerator_validation.require_nonzero_gpu_memory_residency // true' "${POLICY_FILE}"
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
accelerator_validation_required=false
if [[ "${expected_execution_backend}" == "cuda" ]]; then
  accelerator_validation_required=true
fi

accelerator_stage_backend=""
accelerator_observability_backend=""
accelerator_backed_declared=false
accelerator_stage_optimizer_steps=0
accelerator_stage_mean_step_latency_ms=0
accelerator_mean_tokens_per_second=0
accelerator_peak_tokens_per_second=0
accelerator_optimizer_steps_completed=0
accelerator_wall_clock_ms=0
if [[ -n "${stage_receipt_file}" ]]; then
  accelerator_stage_backend="$(
    jq -r '.delivered_execution.runtime_backend // empty' "${stage_receipt_file}"
  )"
  accelerator_backed_declared="$(
    jq -r '.accelerator_execution.accelerator_backed // false' "${stage_receipt_file}"
  )"
  accelerator_stage_optimizer_steps="$(
    jq -r '.accelerator_execution.optimizer_steps_completed // 0' "${stage_receipt_file}"
  )"
  accelerator_stage_mean_step_latency_ms="$(
    jq -r '.accelerator_execution.mean_step_latency_ms // 0' "${stage_receipt_file}"
  )"
fi
if [[ -n "${observability_receipt_file}" ]]; then
  accelerator_observability_backend="$(
    jq -r '.hardware_topology.delivered_execution.runtime_backend // empty' \
      "${observability_receipt_file}"
  )"
  accelerator_mean_tokens_per_second="$(
    jq -r '.throughput.mean_tokens_per_second // 0' "${observability_receipt_file}"
  )"
  accelerator_peak_tokens_per_second="$(
    jq -r '.throughput.peak_tokens_per_second // 0' "${observability_receipt_file}"
  )"
  accelerator_optimizer_steps_completed="$(
    jq -r '.throughput.optimizer_steps_completed // 0' "${observability_receipt_file}"
  )"
  accelerator_wall_clock_ms="$(
    jq -r '.throughput.wall_clock_ms // 0' "${observability_receipt_file}"
  )"
fi

accelerator_gpu_stats_json="$(
  awk -F',' -v warmup_samples="${warmup_samples}" '
    BEGIN {
      raw_samples = 0
      considered = 0
      nonzero_util = 0
      nonzero_mem = 0
      sum_gpu = 0
      max_gpu = 0
      max_mem = 0
      observed_mem_total = 0
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
      mem_total = $6
      gsub(/^[ \t]+|[ \t]+$/, "", gpu_util)
      gsub(/^[ \t]+|[ \t]+$/, "", mem_used)
      gsub(/^[ \t]+|[ \t]+$/, "", mem_total)
      gpu_util += 0
      mem_used += 0
      mem_total += 0
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
      observed_mem_total = mem_total
    }
    END {
      avg_gpu = considered > 0 ? sum_gpu / considered : 0
      printf("{\"considered_samples\":%d,\"nonzero_utilization_samples\":%d,\"nonzero_memory_samples\":%d,\"avg_gpu_util_percent\":%.2f,\"max_gpu_util_percent\":%.2f,\"max_memory_used_mib\":%.2f,\"observed_memory_total_mib\":%.2f}", considered, nonzero_util, nonzero_mem, avg_gpu, max_gpu, max_mem, observed_mem_total)
    }
  ' "${gpu_samples_path}" 2>/dev/null || printf '%s' '{"considered_samples":0,"nonzero_utilization_samples":0,"nonzero_memory_samples":0,"avg_gpu_util_percent":0,"max_gpu_util_percent":0,"max_memory_used_mib":0,"observed_memory_total_mib":0}'
)"
accelerator_considered_samples="$(jq -r '.considered_samples' <<<"${accelerator_gpu_stats_json}")"
accelerator_nonzero_utilization_samples="$(
  jq -r '.nonzero_utilization_samples' <<<"${accelerator_gpu_stats_json}"
)"
accelerator_nonzero_memory_samples="$(
  jq -r '.nonzero_memory_samples' <<<"${accelerator_gpu_stats_json}"
)"

accelerator_failure_reasons=()
if [[ "${accelerator_validation_required}" == "true" ]]; then
  if [[ -z "${stage_receipt_file}" ]]; then
    accelerator_failure_reasons+=("missing_stage_receipt")
  fi
  if [[ -z "${observability_receipt_file}" ]]; then
    accelerator_failure_reasons+=("missing_observability_receipt")
  fi
  if [[ "${accelerator_stage_backend}" != "cuda" ]]; then
    accelerator_failure_reasons+=("stage_backend_not_cuda")
  fi
  if [[ -n "${observability_receipt_file}" && "${accelerator_observability_backend}" != "cuda" ]]; then
    accelerator_failure_reasons+=("observability_backend_not_cuda")
  fi
  if [[ "${accelerator_backed_declared}" != "true" ]]; then
    accelerator_failure_reasons+=("accelerator_execution_not_declared")
  fi
  if (( accelerator_considered_samples < minimum_post_warmup_samples )); then
    accelerator_failure_reasons+=("insufficient_post_warmup_gpu_samples")
  fi
  if [[ "${require_nonzero_gpu_utilization}" == "true" ]] && (( accelerator_nonzero_utilization_samples == 0 )); then
    accelerator_failure_reasons+=("zero_post_warmup_gpu_utilization")
  fi
  if [[ "${require_nonzero_gpu_memory_residency}" == "true" ]] && (( accelerator_nonzero_memory_samples == 0 )); then
    accelerator_failure_reasons+=("zero_post_warmup_gpu_memory_residency")
  fi
  if (( accelerator_mean_tokens_per_second == 0 )); then
    accelerator_failure_reasons+=("zero_mean_tokens_per_second")
  fi
  if (( accelerator_optimizer_steps_completed == 0 )); then
    accelerator_failure_reasons+=("zero_optimizer_steps_completed")
  fi
fi
if (( ${#accelerator_failure_reasons[@]} == 0 )); then
  accelerator_failure_reasons_json='[]'
else
  accelerator_failure_reasons_json="$(
    printf '%s\n' "${accelerator_failure_reasons[@]}" | jq -R . | jq -s '.'
  )"
fi
accelerator_backed_pass=true
if [[ "${accelerator_validation_required}" == "true" && "${accelerator_failure_reasons_json}" != "[]" ]]; then
  accelerator_backed_pass=false
fi
accelerator_validation_detail="$(
  if [[ "${accelerator_validation_required}" != "true" ]]; then
    printf 'Launch manifest expected backend `%s`; accelerator validation was not required for this lane.' "${expected_execution_backend}"
  elif [[ "${accelerator_backed_pass}" == "true" ]]; then
    printf 'Trainer lane `%s` satisfied cuda backend, non-zero post-warmup GPU utilization, non-zero GPU memory residency, and non-zero throughput evidence.' "${trainer_lane_id}"
  else
    printf 'Trainer lane `%s` failed accelerator validation for reasons: %s.' "${trainer_lane_id}" "$(jq -r 'join(", ")' <<<"${accelerator_failure_reasons_json}")"
  fi
)"
if [[ "${accelerator_validation_required}" == "true" && "${accelerator_backed_pass}" != "true" && "${final_result_classification}" == "bounded_success" ]]; then
  final_result_classification="training_runtime_failure"
  if [[ -z "${final_failure_detail}" ]]; then
    final_failure_detail="${accelerator_validation_detail}"
  fi
fi

jq -n \
  --arg schema_version "psion.google_accelerator_validation_receipt.v1" \
  --arg created_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --arg trainer_lane_id "${trainer_lane_id}" \
  --arg expected_execution_backend "${expected_execution_backend}" \
  --arg observed_stage_execution_backend "${accelerator_stage_backend}" \
  --arg observed_observability_execution_backend "${accelerator_observability_backend}" \
  --arg stage_receipt_path "${stage_receipt_file}" \
  --arg observability_receipt_path "${observability_receipt_file}" \
  --argjson accelerator_validation_required "${accelerator_validation_required}" \
  --argjson accelerator_backed_declared "${accelerator_backed_declared}" \
  --argjson accelerator_backed_pass "${accelerator_backed_pass}" \
  --argjson sample_interval_seconds "${sample_interval_seconds}" \
  --argjson warmup_seconds "${warmup_seconds}" \
  --argjson warmup_samples_skipped "${warmup_samples}" \
  --argjson minimum_post_warmup_samples "${minimum_post_warmup_samples}" \
  --argjson stage_optimizer_steps "${accelerator_stage_optimizer_steps}" \
  --argjson stage_mean_step_latency_ms "${accelerator_stage_mean_step_latency_ms}" \
  --argjson mean_tokens_per_second "${accelerator_mean_tokens_per_second}" \
  --argjson peak_tokens_per_second "${accelerator_peak_tokens_per_second}" \
  --argjson optimizer_steps_completed "${accelerator_optimizer_steps_completed}" \
  --argjson wall_clock_ms "${accelerator_wall_clock_ms}" \
  --argjson gpu_summary "$(cat "${gpu_summary_path}")" \
  --argjson gpu_post_warmup_stats "${accelerator_gpu_stats_json}" \
  --argjson failure_reasons "${accelerator_failure_reasons_json}" \
  --arg detail "${accelerator_validation_detail}" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    trainer_lane_id: $trainer_lane_id,
    expected_execution_backend: $expected_execution_backend,
    observed_stage_execution_backend: (if $observed_stage_execution_backend == "" then null else $observed_stage_execution_backend end),
    observed_observability_execution_backend: (if $observed_observability_execution_backend == "" then null else $observed_observability_execution_backend end),
    stage_receipt_path: (if $stage_receipt_path == "" then null else $stage_receipt_path end),
    observability_receipt_path: (if $observability_receipt_path == "" then null else $observability_receipt_path end),
    accelerator_validation_required: $accelerator_validation_required,
    accelerator_backed_declared: $accelerator_backed_declared,
    accelerator_backed_pass: $accelerator_backed_pass,
    sampling_policy: {
      sample_interval_seconds: $sample_interval_seconds,
      warmup_seconds: $warmup_seconds,
      warmup_samples_skipped: $warmup_samples_skipped,
      minimum_post_warmup_samples: $minimum_post_warmup_samples
    },
    stage_metrics: {
      optimizer_steps_completed: $stage_optimizer_steps,
      mean_step_latency_ms: $stage_mean_step_latency_ms
    },
    throughput_metrics: {
      mean_tokens_per_second: $mean_tokens_per_second,
      peak_tokens_per_second: $peak_tokens_per_second,
      optimizer_steps_completed: $optimizer_steps_completed,
      wall_clock_ms: $wall_clock_ms
    },
    gpu_summary: $gpu_summary,
    gpu_post_warmup_stats: $gpu_post_warmup_stats,
    failure_reasons: $failure_reasons,
    detail: $detail
  }' > "${accelerator_validation_path}"

jq -n \
  --arg schema_version "psion.google_run_timeline.v1" \
  --arg run_id "${RUN_ID}" \
  --arg launch_created_at_utc "${launch_created_at_utc}" \
  --arg bootstrap_started_at_utc "${BOOTSTRAP_STARTED_AT_UTC}" \
  --arg bootstrap_finished_at_utc "${BOOTSTRAP_FINISHED_AT_UTC}" \
  --arg training_started_at_utc "${TRAINING_STARTED_AT_UTC}" \
  --arg training_finished_at_utc "${TRAINING_FINISHED_AT_UTC}" \
  --arg checkpoint_completed_at_utc "${CHECKPOINT_COMPLETED_AT_UTC}" \
  --arg teardown_started_at_utc "${TEARDOWN_STARTED_AT_UTC}" \
  --arg teardown_finished_at_utc "$(timestamp_utc)" \
  '{
    schema_version: $schema_version,
    run_id: $run_id,
    timestamps_utc: {
      launch_created_at_utc: $launch_created_at_utc,
      bootstrap_started_at_utc: (if $bootstrap_started_at_utc == "" then null else $bootstrap_started_at_utc end),
      bootstrap_finished_at_utc: (if $bootstrap_finished_at_utc == "" then null else $bootstrap_finished_at_utc end),
      training_started_at_utc: (if $training_started_at_utc == "" then null else $training_started_at_utc end),
      training_finished_at_utc: (if $training_finished_at_utc == "" then null else $training_finished_at_utc end),
      checkpoint_completed_at_utc: (if $checkpoint_completed_at_utc == "" then null else $checkpoint_completed_at_utc end),
      teardown_started_at_utc: (if $teardown_started_at_utc == "" then null else $teardown_started_at_utc end),
      teardown_finished_at_utc: $teardown_finished_at_utc
    }
  }' > "${timeline_path}"

jq -n \
  --arg schema_version "psion.google_run_outcome.v1" \
  --arg created_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --arg failure_code "${final_result_classification}" \
  --arg failure_detail "${final_failure_detail}" \
  --arg training_exit_code "${TRAINING_EXIT_CODE}" \
  --arg archive_manifest_uri "${ARCHIVE_MANIFEST_URI}" \
  --arg cold_restore_manifest_uri "${COLD_RESTORE_MANIFEST_URI}" \
  --argjson accelerator_validation "$(cat "${accelerator_validation_path}")" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    result_classification: $failure_code,
    training_exit_code: ($training_exit_code | tonumber),
    failure_detail: (if $failure_detail == "" then null else $failure_detail end),
    archive_manifest_uri: (if $archive_manifest_uri == "" then null else $archive_manifest_uri end),
    cold_restore_manifest_uri: (if $cold_restore_manifest_uri == "" then null else $cold_restore_manifest_uri end),
    accelerator_validation: $accelerator_validation
  }' > "${outcome_path}"

collect_remote_sha_record "launch_manifest" "manifest" "${LAUNCH_MANIFEST_URI}" "${tmpdir}/launch_reference.json"
collect_remote_sha_record "startup_script_snapshot" "artifact" "${startup_script_uri}" "${tmpdir}/startup_script_snapshot.sh"
collect_remote_sha_record "quota_preflight" "manifest" "${preflight_uri}" "${tmpdir}/quota_preflight.json"
collect_remote_sha_record "input_package_descriptor" "manifest" "${input_descriptor_uri}" "${tmpdir}/input_package_descriptor.json"
record_reference_object "input_package_archive" "artifact" "${input_archive_uri}" "${input_archive_sha256}" ""

upload_local_artifact "host_facts" "manifest" "${host_facts_path}" "${host_prefix}/${host_facts_name}"
upload_local_artifact "runtime_snapshot" "manifest" "${runtime_snapshot_path}" "${host_prefix}/${runtime_snapshot_name}"
upload_local_artifact "run_timeline" "manifest" "${timeline_path}" "${host_prefix}/${timeline_name}"
upload_local_artifact "run_outcome" "manifest" "${outcome_path}" "${host_prefix}/${outcome_name}"
upload_local_artifact "accelerator_validation_receipt" "receipt" "${accelerator_validation_path}" "${receipts_prefix}/${accelerator_validation_receipt_name}"

if [[ -f "${event_log_path}" ]]; then
  upload_local_artifact "run_events" "event_log" "${event_log_path}" "${logs_prefix}/${event_log_name}"
fi
if [[ -f "${LOG_FILE:-/var/log/psion-google-startup.log}" ]]; then
  upload_local_artifact "startup_stdout_stderr" "log" "${LOG_FILE:-/var/log/psion-google-startup.log}" "${logs_prefix}/psion-google-startup.log"
fi
if [[ -f "${gpu_samples_path}" ]]; then
  upload_local_artifact "gpu_samples" "metrics" "${gpu_samples_path}" "${logs_prefix}/${gpu_samples_name}"
fi
upload_local_artifact "gpu_summary" "manifest" "${gpu_summary_path}" "${host_prefix}/${gpu_summary_name}"

if [[ -d "${output_dir}" ]]; then
  while IFS= read -r local_path; do
    rel_path="${local_path#${output_dir}/}"
    role="$(classify_local_role "$(basename "${local_path}")")"
    if [[ "$(basename "${local_path}")" == *.safetensors ]]; then
      remote_uri="${artifacts_prefix}/${rel_path}"
      artifact_kind="reference_pilot_checkpoint"
    else
      remote_uri="${receipts_prefix}/${rel_path}"
      artifact_kind="${rel_path}"
    fi
    upload_local_artifact "${artifact_kind}" "${role}" "${local_path}" "${remote_uri}"
  done < <(find "${output_dir}" -type f | sort)
fi

if [[ -d "${log_dir}" ]]; then
  while IFS= read -r local_path; do
    base_name="$(basename "${local_path}")"
    case "${base_name}" in
      "${event_log_name}"|"${gpu_samples_name}")
        continue
        ;;
    esac
    upload_local_artifact "${base_name}" "$(classify_local_role "${base_name}")" "${local_path}" "${logs_prefix}/${base_name}"
  done < <(find "${log_dir}" -maxdepth 1 -type f | sort)
fi

if [[ -n "${ARCHIVE_MANIFEST_URI}" ]]; then
  collect_remote_sha_record "checkpoint_archive_manifest" "manifest" "${ARCHIVE_MANIFEST_URI}" "${tmpdir}/checkpoint_archive_manifest.json"
fi
if [[ -n "${COLD_RESTORE_MANIFEST_URI}" ]]; then
  cold_restore_local="${tmpdir}/cold_restore_manifest.json"
  collect_remote_sha_record "cold_restore_manifest" "manifest" "${COLD_RESTORE_MANIFEST_URI}" "${cold_restore_local}"
  resume_probe_uri="$(jq -r '.resume_probe_uri // empty' "${cold_restore_local}")"
  if [[ -n "${resume_probe_uri}" ]]; then
    collect_remote_sha_record "cold_restore_resume_probe" "manifest" "${resume_probe_uri}" "${tmpdir}/cold_restore_resume_probe.json"
  fi
fi

manifest_of_manifests_file="${tmpdir}/${manifest_of_manifests_name}"
jq -n \
  --arg schema_version "psion.google_run_manifest_of_manifests.v1" \
  --arg created_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --argjson manifests "$(jq -s 'map(select(.evidence_role == "manifest" or .evidence_role == "receipt"))' "${artifact_records_file}")" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    manifests: $manifests
  }' > "${manifest_of_manifests_file}"

manifest_of_manifests_uri="${final_prefix}/${manifest_of_manifests_name}"
manifest_of_manifests_sha256="$(compute_sha256 "${manifest_of_manifests_file}")"
upload_local_artifact "manifest_of_manifests" "manifest" "${manifest_of_manifests_file}" "${manifest_of_manifests_uri}"
artifacts_json="$(jq -s '.' "${artifact_records_file}")"

final_manifest_file="${tmpdir}/${final_manifest_name}"
jq -n \
  --arg schema_version "psion.google_run_final_manifest.v1" \
  --arg created_at_utc "$(timestamp_utc)" \
  --arg run_id "${RUN_ID}" \
  --arg project_id "${project_id}" \
  --arg zone "${zone}" \
  --arg instance_name "${instance_name}" \
  --arg profile_id "${profile_id}" \
  --arg profile_label "${profile_label}" \
  --arg machine_type "${machine_type}" \
  --arg accelerator_type "${accelerator_type}" \
  --argjson accelerator_count "${accelerator_count}" \
  --arg result_classification "${final_result_classification}" \
  --arg failure_detail "${final_failure_detail}" \
  --arg launch_manifest_uri "${LAUNCH_MANIFEST_URI}" \
  --arg startup_script_uri "${startup_script_uri}" \
  --arg quota_preflight_uri "${preflight_uri}" \
  --arg input_descriptor_uri "${input_descriptor_uri}" \
  --arg input_archive_uri "${input_archive_uri}" \
  --arg input_archive_sha256 "${input_archive_sha256}" \
  --arg input_manifest_sha256 "${input_manifest_sha256}" \
  --arg archive_manifest_uri "${ARCHIVE_MANIFEST_URI}" \
  --arg cold_restore_manifest_uri "${COLD_RESTORE_MANIFEST_URI}" \
  --arg manifest_of_manifests_uri "${manifest_of_manifests_uri}" \
  --arg manifest_of_manifests_sha256 "${manifest_of_manifests_sha256}" \
  --argjson timeline "$(cat "${timeline_path}")" \
  --argjson gpu_summary "$(cat "${gpu_summary_path}")" \
  --argjson accelerator_validation "$(cat "${accelerator_validation_path}")" \
  --argjson outcome "$(cat "${outcome_path}")" \
  --argjson retained_objects "${artifacts_json}" \
  '{
    schema_version: $schema_version,
    created_at_utc: $created_at_utc,
    run_id: $run_id,
    project_id: $project_id,
    topology: {
      zone: $zone,
      instance_name: $instance_name,
      profile_id: $profile_id,
      profile_label: $profile_label,
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: $accelerator_count
    },
    result_classification: $result_classification,
    failure_detail: (if $failure_detail == "" then null else $failure_detail end),
    launch_artifacts: {
      launch_manifest_uri: $launch_manifest_uri,
      startup_script_uri: $startup_script_uri,
      quota_preflight_uri: $quota_preflight_uri
    },
    input_package: {
      descriptor_uri: $input_descriptor_uri,
      archive_uri: $input_archive_uri,
      archive_sha256: $input_archive_sha256,
      manifest_sha256: $input_manifest_sha256
    },
    checkpoint_recovery: {
      archive_manifest_uri: (if $archive_manifest_uri == "" then null else $archive_manifest_uri end),
      cold_restore_manifest_uri: (if $cold_restore_manifest_uri == "" then null else $cold_restore_manifest_uri end)
    },
    timeline: $timeline,
    gpu_summary: $gpu_summary,
    accelerator_validation: $accelerator_validation,
    outcome: $outcome,
    manifest_of_manifests: {
      remote_uri: $manifest_of_manifests_uri,
      sha256: $manifest_of_manifests_sha256
    },
    retained_objects: $retained_objects,
    detail: "Google single-node Psion run folder preserves launch truth, host facts, accelerator validation, training receipts, checkpoint recovery receipts, logs, and per-object digests for postmortem audit."
  }' > "${final_manifest_file}"

final_manifest_uri="${final_prefix}/${final_manifest_name}"
gcloud storage cp --quiet "${final_manifest_file}" "${final_manifest_uri}" >/dev/null
wait_for_object "${final_manifest_uri}"

echo "run final manifest:"
cat "${final_manifest_file}"

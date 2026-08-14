#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DEFAULT_MODEL="${REPO_ROOT}/target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf"
readonly MODEL_PATH="${1:-${DEFAULT_MODEL}}"
readonly OUTPUT_DIR="${2:-${REPO_ROOT}/fixtures/qwen38/reports}"
readonly BENCH_BIN="${REPO_ROOT}/target/release/examples/qwen35_cuda_bench"
readonly IDLE_QUERY="nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits"
readonly EXPECTED_ARTIFACT_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly EXPECTED_ARTIFACT_BYTES=13441059904

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Qwen3.8 GGUF not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

readonly ARTIFACT_BYTES="$(stat -c '%s' "${MODEL_PATH}")"
readonly ARTIFACT_SHA256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
if [[ "${ARTIFACT_BYTES}" != "${EXPECTED_ARTIFACT_BYTES}" ]]; then
  echo "Qwen3.8 GGUF byte-length mismatch: expected ${EXPECTED_ARTIFACT_BYTES}, actual ${ARTIFACT_BYTES}" >&2
  exit 1
fi
if [[ "${ARTIFACT_SHA256}" != "${EXPECTED_ARTIFACT_SHA256}" ]]; then
  echo "Qwen3.8 GGUF SHA-256 mismatch: expected ${EXPECTED_ARTIFACT_SHA256}, actual ${ARTIFACT_SHA256}" >&2
  exit 1
fi

(
  cd "${REPO_ROOT}"
  cargo build --release -p psionic-serve --example qwen35_cuda_bench
)

gpu_identity() {
  nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv,noheader,nounits |
    head -n 1
}

require_idle_gpu() {
  local compute_processes
  compute_processes="$(nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits)"
  if [[ -n "${compute_processes//[[:space:]]/}" ]]; then
    echo "refusing Qwen3.8 CUDA evidence run because the GPU is not idle:" >&2
    printf '%s\n' "${compute_processes}" >&2
    exit 2
  fi
  printf '%s' "${compute_processes}"
}

run_evidence() {
  local mode="$1"
  local output_path="$2"
  shift 2

  local idle_output
  local checked_at_unix_s
  local gpu_row
  local raw_report
  idle_output="$(require_idle_gpu)"
  checked_at_unix_s="$(date +%s)"
  gpu_row="$(gpu_identity)"
  raw_report="$(mktemp)"
  trap 'rm -f "${raw_report}"' RETURN

  "${BENCH_BIN}" \
    --backend psionic \
    --model-path "${MODEL_PATH}" \
    --prompt Hello \
    --raw-prompt \
    --max-output-tokens 2 \
    --repeats 2 \
    "$@" \
    --json-out "${raw_report}"

  jq \
    --arg mode "${mode}" \
    --arg idle_command "${IDLE_QUERY}" \
    --arg idle_output "${idle_output}" \
    --argjson checked_at_unix_s "${checked_at_unix_s}" \
    --arg gpu_row "${gpu_row}" \
    --arg artifact_sha256 "${ARTIFACT_SHA256}" \
    --argjson artifact_bytes "${ARTIFACT_BYTES}" \
    '
      .schema_version = 7
      | .report_kind = "qwen38_cuda_generation_evidence"
      | .qwen38_evidence = {
          phase: "R7",
          mode: $mode,
          artifact: {
            filename: "Qwen3.8-27B-UD-Q3_K_XL.gguf",
            byte_length: $artifact_bytes,
            sha256: $artifact_sha256
          },
          gpu_idle_check: {
            command: $idle_command,
            checked_at_unix_s: $checked_at_unix_s,
            output: $idle_output,
            idle: ($idle_output == "")
          },
          gpu_identity_csv: $gpu_row,
          reference: {
            implementation: "ggml-org/llama.cpp",
            revision: "9b05354ec6fb58b4e665e9a39ebc40285c015638",
            prompt: "Hello",
            prompt_token_ids: [9419],
            greedy_output_token_ids: [11, 353]
          },
          residency_measurement: {
            status: "measured_psionic_cuda_allocator_high_water_inside_live_preflight_envelope",
            scope: .psionic_cuda_startup.allocator_measurement_scope,
            planned_peak_device_bytes: .psionic_cuda_startup.runtime_contract.planned_device_bytes,
            allocator_peak_resident_device_bytes: .psionic_cuda_startup.allocator_peak_resident_device_bytes_after_measurements,
            allocator_resident_device_bytes_after_measurements: .psionic_cuda_startup.allocator_resident_device_bytes_after_measurements,
            preflight_free_device_bytes: .psionic_cuda_startup.runtime_contract.device_free_bytes_at_preflight,
            preflight_total_device_bytes: .psionic_cuda_startup.runtime_contract.device_capacity_bytes
          },
          validation_scope: {
            context_tokens: 4096,
            larger_context_evidence: "not_claimed",
            server_publication: "deferred_to_R8"
          }
        }
    ' "${raw_report}" >"${output_path}"
  rm -f "${raw_report}"
  trap - RETURN
}

run_evidence \
  greedy \
  "${OUTPUT_DIR}/qwen38_cuda_greedy_generation_v1.json" \
  --decode greedy \
  --require-fallback-free-cuda

run_evidence \
  bounded_sample \
  "${OUTPUT_DIR}/qwen38_cuda_bounded_sample_generation_v1.json" \
  --decode sample \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --seed 42

"${REPO_ROOT}/scripts/check-qwen38-cuda-generation.sh" "${OUTPUT_DIR}"

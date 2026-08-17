#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODEL_DIR="${PSIONIC_QWEN38_OFFICIAL_MODEL_DIR:-${ROOT}/target/models/qwen/Qwen3.8-27B}"
readonly DEFAULT_GGUF="${ROOT}/target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf"
readonly GGUF_PATH="${PSIONIC_QWEN38_PILOT_GGUF_PATH:-${DEFAULT_GGUF}}"
readonly REPORT_PATH="${PSIONIC_QWEN38_MULTIMODAL_CUDA_REPORT_PATH:-${ROOT}/fixtures/qwen38/reports/qwen38_multimodal_cuda_evidence_v1.json}"
readonly SMOKE_BIN="${ROOT}/target/release/examples/qwen38_multimodal_cuda_smoke"
readonly SOURCE_SHARD="${MODEL_DIR}/model-00001-of-00018.safetensors"
readonly EXPECTED_OFFICIAL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
readonly EXPECTED_SOURCE_SHARD_BYTES=3966730552
readonly EXPECTED_SOURCE_SHARD_SHA256="ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
readonly EXPECTED_GGUF_BYTES=13441059904
readonly EXPECTED_GGUF_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly IMAGE_PARITY_REPORT="${ROOT}/fixtures/qwen38/reports/qwen38_vision_parity_v1.json"
readonly VIDEO_PARITY_REPORT="${ROOT}/fixtures/qwen38/reports/qwen38_vision_video_parity_v1.json"
readonly IDLE_COMMAND="nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits"

cd "${ROOT}"

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "Qwen3.8 multimodal CUDA evidence requires branch main" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Qwen3.8 multimodal CUDA evidence requires a clean checkout" >&2
  exit 1
fi
git fetch origin main --quiet
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "Qwen3.8 multimodal CUDA evidence requires HEAD equal to origin/main" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_SHARD}" || ! -f "${GGUF_PATH}" ]]; then
  echo "Qwen3.8 multimodal CUDA evidence artifacts are missing" >&2
  exit 1
fi
if [[ "$(stat -c %s "${SOURCE_SHARD}")" != "${EXPECTED_SOURCE_SHARD_BYTES}" ]]; then
  echo "Qwen3.8 vision source shard byte-length mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "${SOURCE_SHARD}" | cut -d' ' -f1)" != "${EXPECTED_SOURCE_SHARD_SHA256}" ]]; then
  echo "Qwen3.8 vision source shard SHA-256 mismatch" >&2
  exit 1
fi
if [[ "$(stat -c %s "${GGUF_PATH}")" != "${EXPECTED_GGUF_BYTES}" ]]; then
  echo "Qwen3.8 decoder GGUF byte-length mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "${GGUF_PATH}" | cut -d' ' -f1)" != "${EXPECTED_GGUF_SHA256}" ]]; then
  echo "Qwen3.8 decoder GGUF SHA-256 mismatch" >&2
  exit 1
fi

cargo test -p psionic-models qwen38_multimodal --lib
cargo test -p psionic-backend-cuda tests::cuda_submission_f16_kv_mrope_uses_each_interleaved_axis_when_available --lib -- --exact --nocapture
cargo build --release -p psionic-serve --features qwen38-vision-cuda --example qwen38_multimodal_cuda_smoke

readonly TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

require_idle_gpu() {
  local processes
  processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits)"
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    echo "refusing Qwen3.8 multimodal CUDA evidence because the GPU is not idle:" >&2
    printf '%s\n' "${processes}" >&2
    exit 2
  fi
  printf '%s' "${processes}"
}

run_row() {
  local media_kind="$1"
  local row_path="${TMP_DIR}/${media_kind}.json"
  local raw_path="${TMP_DIR}/${media_kind}-raw.json"
  local idle_output
  local checked_at_unix_s

  idle_output="$(require_idle_gpu)"
  checked_at_unix_s="$(date +%s)"
  "${SMOKE_BIN}" "${MODEL_DIR}" "${GGUF_PATH}" "${media_kind}" >"${raw_path}"
  jq \
    --arg command "${IDLE_COMMAND}" \
    --arg output "${idle_output}" \
    --argjson checked_at_unix_s "${checked_at_unix_s}" \
    '. + {
      gpu_idle_check: {
        command: $command,
        checked_at_unix_s: $checked_at_unix_s,
        output: $output,
        idle: ($output == "")
      }
    }' "${raw_path}" >"${row_path}"
}

run_row image
run_row video

mkdir -p "$(dirname "${REPORT_PATH}")"
jq -n \
  --arg revision "$(git rev-parse HEAD)" \
  --arg official_revision "${EXPECTED_OFFICIAL_REVISION}" \
  --arg source_shard_sha256 "${EXPECTED_SOURCE_SHARD_SHA256}" \
  --argjson source_shard_bytes "${EXPECTED_SOURCE_SHARD_BYTES}" \
  --arg gguf_sha256 "${EXPECTED_GGUF_SHA256}" \
  --argjson gguf_bytes "${EXPECTED_GGUF_BYTES}" \
  --arg image_parity_sha256 "$(sha256sum "${IMAGE_PARITY_REPORT}" | cut -d' ' -f1)" \
  --arg video_parity_sha256 "$(sha256sum "${VIDEO_PARITY_REPORT}" | cut -d' ' -f1)" \
  --arg gpu_name "$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)" \
  --arg driver_version "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)" \
  --slurpfile image "${TMP_DIR}/image.json" \
  --slurpfile video "${TMP_DIR}/video.json" \
  '{
    schema_version: "psionic.qwen38.multimodal_cuda_evidence.v1",
    status: "implemented_early",
    phase: "R11",
    psionic_revision: $revision,
    artifacts: {
      official_model_revision: $official_revision,
      vision_source_shard: {
        filename: "model-00001-of-00018.safetensors",
        byte_length: $source_shard_bytes,
        sha256: $source_shard_sha256
      },
      decoder_gguf: {
        filename: "Qwen3.8-27B-UD-Q3_K_XL.gguf",
        byte_length: $gguf_bytes,
        sha256: $gguf_sha256
      }
    },
    linked_encoder_parity: {
      image_report: "fixtures/qwen38/reports/qwen38_vision_parity_v1.json",
      image_report_sha256: $image_parity_sha256,
      video_report: "fixtures/qwen38/reports/qwen38_vision_video_parity_v1.json",
      video_report_sha256: $video_parity_sha256
    },
    gpu: {
      name: $gpu_name,
      driver_version: $driver_version
    },
    rows: [$image[0], $video[0]],
    claim_boundary: {
      cuda_image_generation: true,
      cuda_video_generation: true,
      metal_decoder_integration: false,
      openai_media_serving: false,
      performance_claim: false
    }
  }' >"${REPORT_PATH}"

"${ROOT}/scripts/check-qwen38-multimodal-cuda.sh" "${REPORT_PATH}"
echo "${REPORT_PATH}"

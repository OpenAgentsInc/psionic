#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODEL_DIR="${PSIONIC_QWEN38_OFFICIAL_MODEL_DIR:-${ROOT}/target/models/qwen/Qwen3.8-27B}"
readonly DEFAULT_GGUF="${ROOT}/target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf"
readonly GGUF_PATH="${PSIONIC_QWEN38_PILOT_GGUF_PATH:-${DEFAULT_GGUF}}"
readonly REPORT_PATH="${PSIONIC_QWEN38_OPENAI_MEDIA_REPORT_PATH:-${ROOT}/fixtures/qwen38/reports/qwen38_openai_media_evidence_v1.json}"
readonly SERVER_BIN="${ROOT}/target/release/psionic-openai-server"
readonly CLIENT_BIN="${ROOT}/target/release/examples/qwen38_openai_media_smoke"
readonly PORT="${PSIONIC_QWEN38_OPENAI_MEDIA_PORT:-18083}"
readonly BASE_URL="http://127.0.0.1:${PORT}"
readonly SOURCE_SHARD="${MODEL_DIR}/model-00001-of-00018.safetensors"
readonly EXPECTED_OFFICIAL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
readonly EXPECTED_SOURCE_SHARD_BYTES=3966730552
readonly EXPECTED_SOURCE_SHARD_SHA256="ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
readonly EXPECTED_GGUF_BYTES=13441059904
readonly EXPECTED_GGUF_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly IDLE_COMMAND="nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits"

cd "${ROOT}"

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "Qwen3.8 OpenAI media evidence requires branch main" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Qwen3.8 OpenAI media evidence requires a clean checkout" >&2
  exit 1
fi
git fetch origin main --quiet
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "Qwen3.8 OpenAI media evidence requires HEAD equal to origin/main" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_SHARD}" || ! -f "${GGUF_PATH}" ]]; then
  echo "Qwen3.8 OpenAI media evidence artifacts are missing" >&2
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
if [[ -n "$(ss -ltnH "sport = :${PORT}")" ]]; then
  echo "Qwen3.8 OpenAI media evidence port ${PORT} is already in use" >&2
  exit 1
fi

cargo test -p psionic-serve qwen38_ --lib
cargo build --release -p psionic-serve --bin psionic-openai-server --example qwen38_openai_media_smoke

readonly TMP_DIR="$(mktemp -d)"
readonly RAW_REPORT="${TMP_DIR}/raw-report.json"
readonly SERVER_LOG="${TMP_DIR}/server.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

idle_output="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits)"
if [[ -n "${idle_output//[[:space:]]/}" ]]; then
  echo "refusing Qwen3.8 OpenAI media evidence because the GPU is not idle:" >&2
  printf '%s\n' "${idle_output}" >&2
  exit 2
fi
readonly idle_output
readonly idle_checked_at_unix_s="$(date +%s)"

"${SERVER_BIN}" \
  -m "${GGUF_PATH}" \
  --backend cuda \
  --qwen38-vision-model-dir "${MODEL_DIR}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

server_ready=false
for _ in $(seq 1 300); do
  if curl --silent --fail "${BASE_URL}/health" >/dev/null 2>&1; then
    server_ready=true
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Qwen3.8 OpenAI media server exited during startup" >&2
    tail -n 100 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 1
done
if [[ "${server_ready}" != "true" ]]; then
  echo "Qwen3.8 OpenAI media server did not become ready" >&2
  tail -n 100 "${SERVER_LOG}" >&2
  exit 1
fi

"${CLIENT_BIN}" "${BASE_URL}" >"${RAW_REPORT}"

kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
SERVER_PID=""

mkdir -p "$(dirname "${REPORT_PATH}")"
jq \
  --arg revision "$(git rev-parse HEAD)" \
  --arg official_revision "${EXPECTED_OFFICIAL_REVISION}" \
  --arg source_shard_sha256 "${EXPECTED_SOURCE_SHARD_SHA256}" \
  --argjson source_shard_bytes "${EXPECTED_SOURCE_SHARD_BYTES}" \
  --arg gguf_sha256 "${EXPECTED_GGUF_SHA256}" \
  --argjson gguf_bytes "${EXPECTED_GGUF_BYTES}" \
  --arg gpu_name "$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)" \
  --arg driver_version "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)" \
  --arg idle_command "${IDLE_COMMAND}" \
  --arg idle_output "${idle_output}" \
  --argjson idle_checked_at_unix_s "${idle_checked_at_unix_s}" \
  --arg server_log_sha256 "$(sha256sum "${SERVER_LOG}" | cut -d' ' -f1)" \
  '. + {
    schema_version: "psionic.qwen38.openai_media_evidence.v1",
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
    gpu: {
      name: $gpu_name,
      driver_version: $driver_version,
      idle_check: {
        command: $idle_command,
        checked_at_unix_s: $idle_checked_at_unix_s,
        output: $idle_output,
        idle: ($idle_output == "")
      }
    },
    server_log_sha256: $server_log_sha256
  }' "${RAW_REPORT}" >"${REPORT_PATH}"

"${ROOT}/scripts/check-qwen38-openai-media.sh" "${REPORT_PATH}"
echo "${REPORT_PATH}"

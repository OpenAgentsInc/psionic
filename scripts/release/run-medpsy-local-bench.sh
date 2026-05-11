#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out_dir="${repo_root}/fixtures/medpsy/benchmarks/manual"
mkdir -p "${out_dir}"

run_row() {
  local label="$1"
  local artifact_kind="$2"
  local model_size="$3"
  local model_path="$4"
  local backend="${5:-cpu}"
  local out_path="${out_dir}/${label}.json"

  cargo run --release -p psionic-models --example medpsy_bench -- \
    --model-path "${model_path}" \
    --artifact-kind "${artifact_kind}" \
    --model-size "${model_size}" \
    --backend "${backend}" \
    --prompt-token-ids "151644" \
    --max-new-tokens "1" \
    --repeats "1" \
    --json-out "${out_path}"

  printf 'wrote %s\n' "${out_path}"
}

backend="${PSIONIC_MEDPSY_BENCH_BACKEND:-cpu}"

if [[ -n "${PSIONIC_MEDPSY_17B_SAFETENSORS_PATH:-}" ]]; then
  run_row "medpsy_17b_safetensors_${backend}" "safetensors" "1.7b" "${PSIONIC_MEDPSY_17B_SAFETENSORS_PATH}" "${backend}"
fi

if [[ -n "${PSIONIC_MEDPSY_17B_Q4_K_M_GGUF_PATH:-}" ]]; then
  run_row "medpsy_17b_q4_k_m_gguf_${backend}" "gguf" "1.7b" "${PSIONIC_MEDPSY_17B_Q4_K_M_GGUF_PATH}" "${backend}"
fi

if [[ -z "${PSIONIC_MEDPSY_17B_SAFETENSORS_PATH:-}" && -z "${PSIONIC_MEDPSY_17B_Q4_K_M_GGUF_PATH:-}" ]]; then
  printf 'no MedPsy artifacts configured; set PSIONIC_MEDPSY_17B_SAFETENSORS_PATH or PSIONIC_MEDPSY_17B_Q4_K_M_GGUF_PATH\n' >&2
  exit 2
fi

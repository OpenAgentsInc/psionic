#!/usr/bin/env bash
set -euo pipefail

readonly REPORT_PATH="${1:-fixtures/qwen38/reports/qwen38_speed_gate_evidence_v1.json}"

jq -e '
  .schema_version == "psionic.qwen38.speed_gate_evidence.v1" and
  .phase == "R13" and
  (.source.revision | test("^[0-9a-f]{40}$")) and
  .source.dirty == false and
  .artifact.filename == "Qwen3.8-27B-UD-Q3_K_XL.gguf" and
  .artifact.byte_length == 13441059904 and
  .artifact.sha256 == "00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2" and
  .comparator.implementation == "ggml-org/llama.cpp" and
  (.comparator.revision | test("^[0-9a-f]{40}$")) and
  (.comparator.server.server_version | length > 0) and
  .contract.decode == "greedy" and
  (.contract.repeats >= 3) and
  (.contract.max_output_tokens >= 32) and
  (.host.idle_checks_unix_s | length == 3) and
  .correctness.output_token_parity == true and
  .correctness.psionic_deterministic_runs == true and
  .correctness.llama_cpp_deterministic_runs == true and
  .correctness.known_prefix_tokens == true and
  .correctness.prompt_token_parity == true and
  .correctness.psionic_fallback_free_required == true and
  (.psionic.runs | length == .contract.repeats) and
  (.llama_cpp.runs | length == .contract.repeats) and
  ([.psionic.runs[].termination.classification] | all(. == "max_output_tokens" or . == "eos_token")) and
  ([.llama_cpp.runs[].termination.classification] | all(. == "max_output_tokens" or . == "eos_token")) and
  ([.psionic.runs[].decode_tok_s] | all(. > 0)) and
  ([.llama_cpp.runs[].decode_tok_s] | all(. > 0)) and
  (.psionic.psionic_cuda_startup.warmup_status == "explicit_warmup_completed") and
  (.comparison.psionic_median_decode_tok_s > 0) and
  (.comparison.llama_cpp_median_decode_tok_s > 0) and
  (.comparison.delta_percent | type == "number") and
  (.claim_boundary | length > 0)
' "${REPORT_PATH}" >/dev/null

readonly MEDIAN_P="$(jq -r '.comparison.psionic_median_decode_tok_s' "${REPORT_PATH}")"
readonly MEDIAN_L="$(jq -r '.comparison.llama_cpp_median_decode_tok_s' "${REPORT_PATH}")"
readonly DELTA="$(jq -r '.comparison.delta_percent' "${REPORT_PATH}")"
readonly WINS="$(jq -r '.comparison.psionic_wins' "${REPORT_PATH}")"

echo "qwen38 speed-gate evidence passed: ${REPORT_PATH} (psionic=${MEDIAN_P} tok/s, llama_cpp=${MEDIAN_L} tok/s, delta=${DELTA}%, psionic_wins=${WINS})"

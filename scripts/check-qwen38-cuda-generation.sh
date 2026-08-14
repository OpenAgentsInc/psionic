#!/usr/bin/env bash
set -euo pipefail

readonly REPORT_DIR="${1:-fixtures/qwen38/reports}"
readonly GREEDY_REPORT="${REPORT_DIR}/qwen38_cuda_greedy_generation_v1.json"
readonly SAMPLE_REPORT="${REPORT_DIR}/qwen38_cuda_bounded_sample_generation_v1.json"
readonly EXPECTED_ARTIFACT_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly EXPECTED_ARTIFACT_BYTES=13441059904
readonly EXPECTED_LLAMA_CPP_REVISION="9b05354ec6fb58b4e665e9a39ebc40285c015638"

check_common() {
  local report="$1"
  jq -e \
    --arg artifact_sha256 "${EXPECTED_ARTIFACT_SHA256}" \
    --argjson artifact_bytes "${EXPECTED_ARTIFACT_BYTES}" \
    --arg llama_revision "${EXPECTED_LLAMA_CPP_REVISION}" \
    '
      .schema_version == 7
      and .report_kind == "qwen38_cuda_generation_evidence"
      and .run_status == "ok"
      and .backend == "psionic"
      and .prompt == "Hello"
      and .prompt_mode == "raw_text"
      and .rendered_prompt == "Hello"
      and .max_output_tokens == 2
      and .repeats == 2
      and .qwen38_evidence.phase == "R7"
      and .qwen38_evidence.artifact.filename == "Qwen3.8-27B-UD-Q3_K_XL.gguf"
      and .qwen38_evidence.artifact.byte_length == $artifact_bytes
      and .qwen38_evidence.artifact.sha256 == $artifact_sha256
      and .qwen38_evidence.gpu_idle_check.idle == true
      and .qwen38_evidence.gpu_idle_check.output == ""
      and (.qwen38_evidence.gpu_identity_csv | length) > 0
      and .qwen38_evidence.reference.revision == $llama_revision
      and .qwen38_evidence.reference.prompt_token_ids == [9419]
      and .qwen38_evidence.reference.greedy_output_token_ids == [11, 353]
      and .qwen38_evidence.validation_scope.context_tokens == 4096
      and .qwen38_evidence.validation_scope.larger_context_evidence == "not_claimed"
      and .psionic_cuda_startup.runtime_contract.family == "qwen38"
      and (.psionic_cuda_startup.runtime_contract.artifact_digest | length) == 64
      and .psionic_cuda_startup.runtime_contract.artifact_bytes == $artifact_bytes
      and .psionic_cuda_startup.runtime_contract.context_limit_tokens == 4096
      and .psionic_cuda_startup.runtime_contract.preflight_status == "admitted_before_weight_upload"
      and .psionic_cuda_startup.runtime_contract.preflight_required_device_bytes <= .psionic_cuda_startup.runtime_contract.device_free_bytes_at_preflight
      and .psionic_cuda_startup.runtime_contract.planned_device_bytes <= .psionic_cuda_startup.runtime_contract.preflight_required_device_bytes
      and .psionic_cuda_startup.runtime_contract.weight_device_bytes < .psionic_cuda_startup.runtime_contract.artifact_bytes
      and .psionic_cuda_startup.runtime_contract.dense_f16_mirror_count == 0
      and (.psionic_cuda_startup.runtime_contract.quantization_modes | index("ggml_q3_k")) != null
      and .psionic_cuda_startup.runtime_contract.raw_logits_materialization_observable == true
      and .psionic_cuda_startup.runtime_contract.host_fallback_enabled == false
      and .psionic_cuda_startup.runtime_contract.execution_plan_namespace == "qwen38-native-cuda|v1"
      and (.psionic_cuda_startup.runtime_contract.execution_plan_digest | length) == 64
      and .psionic_cuda_startup.runtime_contract.graph_cache_namespace == "qwen38-cuda-graph-cache|v1"
      and (.psionic_cuda_startup.runtime_contract.graph_cache_identity | length) == 64
      and .qwen38_evidence.residency_measurement.status == "exact_runtime_plan_inside_live_preflight_envelope"
      and .qwen38_evidence.residency_measurement.planned_peak_device_bytes == .psionic_cuda_startup.runtime_contract.planned_device_bytes
      and ([.runs[].qwen35_host_fallback_evidence.fallback_invocations] | add) == 0
      and ([.runs[].qwen35_graph_shape_drifts] | add) == 0
      and ([.runs[].qwen35_graph_hits] | add) > 0
      and ([.runs[].qwen35_graph_cache_identity] | unique) == [.psionic_cuda_startup.runtime_contract.graph_cache_identity]
      and ([.runs[].decode_tok_s] | min) > 0
    ' "${report}" >/dev/null
}

check_common "${GREEDY_REPORT}"
check_common "${SAMPLE_REPORT}"

jq -e '
  .qwen38_evidence.mode == "greedy"
  and .decode_mode == "greedy"
  and .psionic_cuda_fast_path.status == "validated"
  and ([.runs[].output_token_ids] | unique) == [[11, 353]]
  and ([.runs[].qwen35_output_modes] | unique) == [["argmax_only"]]
  and ([.runs[].qwen35_raw_logits] | any) == false
' "${GREEDY_REPORT}" >/dev/null

jq -e '
  .qwen38_evidence.mode == "bounded_sample"
  and .decode_mode == "sample"
  and .temperature == 0.8
  and .top_k == 40
  and .top_p == 0.9
  and .seed == 42
  and ([.runs[].output_token_ids] | unique | length) == 1
  and ([.runs[].qwen35_output_modes] | unique) == [["top_k_candidates:40"]]
  and ([.runs[].qwen35_raw_logits] | any) == false
' "${SAMPLE_REPORT}" >/dev/null

echo "qwen38 CUDA generation evidence passed: ${GREEDY_REPORT}, ${SAMPLE_REPORT}"

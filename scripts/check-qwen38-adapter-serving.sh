#!/usr/bin/env bash
set -euo pipefail

readonly REPORT_PATH="${1:-fixtures/qwen38/reports/qwen38_adapter_serving_evidence_v1.json}"
readonly ARTIFACT_PATH="${2:-fixtures/qwen38/adapters/qwen38_lm_head_lora_real_27b_v1.safetensors}"

jq -e '
  .schema_version == "psionic.qwen38.adapter_serving_evidence.v1" and
  .phase == "R12" and
  .status == "partial" and
  (.source.revision | test("^[0-9a-f]{40}$")) and
  .source.dirty == false and
  .artifact.repository_id == "unsloth/Qwen3.8-27B-GGUF" and
  .artifact.byte_length == 17106775008 and
  .artifact.sha256 == "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169" and
  .decoder_model.config.hidden_size == 5120 and
  (.decoder_model.config.vocab_size > 100000) and
  (.decoder_model.weights_digest | test("^[0-9a-f]{64}$")) and
  (.adapter_artifact.sha256 | test("^[0-9a-f]{64}$")) and
  .adapter_artifact.trained_step == 1 and
  .adapter_artifact.lora_rank == 1 and
  (.adapter_artifact.plan_digest | test("^[0-9a-f]{64}$")) and
  .adapter_artifact.identity[0].base_model_id == "Qwen/Qwen3.8-27B" and
  .adapter_artifact.identity[0].base_model_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" and
  .adapter_artifact.identity[0].artifact_digest == .adapter_artifact.sha256 and
  .adapter_artifact.identity[0].provenance_digest == .adapter_artifact.plan_digest and
  (.baseline.output_token_ids | length == 2) and
  (.adapted.output_token_ids | length == 2) and
  .adapted.first_token_is_target == true and
  .adapted.target_token_id != .baseline.argmax_token_id and
  .adapted.provenance_binding_matches == true and
  .refusals.merged_residency_refused == true and
  .refusals.drifted_binding_refused == true and
  .refusals.detached_binding_refused == true and
  .refusals.detached_binding_matches == true and
  .fallback_policy == "refuse" and
  .hidden_fallback_used == false and
  (.claim_boundary | length > 0)
' "${REPORT_PATH}" >/dev/null

readonly EXPECTED_ARTIFACT_SHA256="$(jq -r '.adapter_artifact.sha256' "${REPORT_PATH}")"
if command -v shasum >/dev/null 2>&1; then
  readonly ACTUAL_ARTIFACT_SHA256="$(shasum -a 256 "${ARTIFACT_PATH}" | awk '{print $1}')"
else
  readonly ACTUAL_ARTIFACT_SHA256="$(sha256sum "${ARTIFACT_PATH}" | awk '{print $1}')"
fi

[[ "${ACTUAL_ARTIFACT_SHA256}" == "${EXPECTED_ARTIFACT_SHA256}" ]]

echo "qwen38 adapter-serving evidence passed: ${REPORT_PATH} ${ARTIFACT_PATH}"

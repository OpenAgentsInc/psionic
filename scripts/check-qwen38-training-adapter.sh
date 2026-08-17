#!/usr/bin/env bash
set -euo pipefail

readonly REPORT_PATH="${1:-fixtures/qwen38/reports/qwen38_training_adapter_evidence_v1.json}"
readonly ARTIFACT_PATH="${2:-fixtures/qwen38/adapters/qwen38_lm_head_lora_tiny_v1.safetensors}"

jq -e '
  .schema_version == "psionic.qwen38.training_adapter_evidence.v1" and
  .phase == "R12" and
  .status == "implemented_early" and
  .base_identity.schema_version == "psionic.qwen38.training_base_identity.v1" and
  .base_identity.model_id == "Qwen/Qwen3.8-27B" and
  .base_identity.served_model_id == "qwen3.8-27b" and
  .base_identity.upstream_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" and
  (.base_identity.artifact_facts_sha256 | test("^[0-9a-f]{64}$")) and
  .base_identity.safetensors_index_sha256 == "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df" and
  (.base_identity.base_artifact_identity_digest | test("^[0-9a-f]{64}$")) and
  .admitted_plan.status == "admitted" and
  .admitted_plan.execution_mode == "tiny_reference_cpu" and
  .admitted_plan.exact_backward_contract == "qwen38_lm_head_lora_f32_reference_backward_v1" and
  .admitted_plan.adapter_binding.base_model_id == "Qwen/Qwen3.8-27B" and
  .admitted_plan.adapter_binding.base_model_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" and
  .admitted_plan.adapter_binding.base_artifact_identity_digest == .base_identity.base_artifact_identity_digest and
  .admitted_plan.adapter_binding.target_modules == ["lm_head.weight"] and
  .admitted_plan.cpu_budget.kind == "bounded_single_core" and
  (.admitted_plan.plan_digest | test("^[0-9a-f]{64}$")) and
  .backward_receipt.schema_version == "psionic.qwen38.lm_head_lora_backward_receipt.v1" and
  .backward_receipt.contract == "qwen38_lm_head_lora_f32_reference_backward_v1" and
  .backward_receipt.hidden_size == 4 and
  .backward_receipt.vocabulary_size == 3 and
  .backward_receipt.lora_rank == 2 and
  .backward_receipt.gradient_check_passed == true and
  .backward_receipt.gradient_max_abs_error <= .backward_receipt.gradient_tolerance and
  .backward_receipt.loss_improved == true and
  .backward_receipt.deterministic_replay == true and
  .backward_receipt.base_weights_frozen == true and
  (.backward_receipt.receipt_digest | test("^[0-9a-f]{64}$")) and
  .checkpoint_recovery.schema_version == "psionic.qwen38.lm_head_lora_recovery_receipt.v1" and
  .checkpoint_recovery.checkpoint.schema_version == "psionic.qwen38.lm_head_lora_checkpoint.v1" and
  .checkpoint_recovery.checkpoint.base_artifact_identity_digest == .base_identity.base_artifact_identity_digest and
  .checkpoint_recovery.checkpoint.adapter_binding_digest == .admitted_plan.adapter_binding.binding_digest and
  .checkpoint_recovery.checkpoint_step == 1 and
  .checkpoint_recovery.resumed_step == 2 and
  .checkpoint_recovery.exact_state_match == true and
  .checkpoint_recovery.optimizer_state_exact_match == true and
  .checkpoint_recovery.second_step_loss_exact_match == true and
  .checkpoint_recovery.tampered_checkpoint_refused == true and
  .checkpoint_recovery.uninterrupted_state_digest == .checkpoint_recovery.resumed_state_digest and
  (.checkpoint_recovery.checkpoint_bytes_sha256 | test("^[0-9a-f]{64}$")) and
  (.checkpoint_recovery.receipt_digest | test("^[0-9a-f]{64}$")) and
  .adapter_artifact.schema_version == "psionic.qwen38.lm_head_lora_adapter_artifact_receipt.v1" and
  .adapter_artifact.artifact_ref == "fixtures/qwen38/adapters/qwen38_lm_head_lora_tiny_v1.safetensors" and
  .adapter_artifact.trained_step == 2 and
  .adapter_artifact.tensor_names == ["lm_head.lora_A.weight", "lm_head.lora_B.weight"] and
  .adapter_artifact.lora_a_shape == [2, 4] and
  .adapter_artifact.lora_b_shape == [3, 2] and
  .adapter_artifact.metadata.schema_version == "psionic.qwen38.lm_head_lora_adapter_artifact_metadata.v1" and
  .adapter_artifact.metadata.base_model_id == "Qwen/Qwen3.8-27B" and
  .adapter_artifact.metadata.base_model_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" and
  .adapter_artifact.metadata.base_artifact_identity_digest == .base_identity.base_artifact_identity_digest and
  .adapter_artifact.metadata.adapter_binding_digest == .admitted_plan.adapter_binding.binding_digest and
  .adapter_artifact.metadata.plan_digest == .admitted_plan.plan_digest and
  .adapter_artifact.metadata.corpus_manifest_sha256 == .admitted_plan.corpus_manifest_sha256 and
  .adapter_artifact.metadata.evaluation_manifest_sha256 == .admitted_plan.evaluation_manifest_sha256 and
  .adapter_artifact.metadata.seed == .admitted_plan.seed and
  .adapter_artifact.metadata.trained_step == 2 and
  .adapter_artifact.metadata.target == "lm_head.weight" and
  .adapter_artifact.metadata.lora_rank == 2 and
  .adapter_artifact.metadata.lora_alpha == 4 and
  .adapter_artifact.adapter_identity.base_model_id == "Qwen/Qwen3.8-27B" and
  .adapter_artifact.adapter_identity.base_model_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" and
  .adapter_artifact.adapter_identity.base_served_artifact_digest == .base_identity.base_artifact_identity_digest and
  .adapter_artifact.adapter_identity.artifact_digest == .adapter_artifact.artifact_sha256 and
  .adapter_artifact.adapter_identity.parameter_count == 14 and
  .adapter_artifact.serving_binding.base_served_artifact_digest == .base_identity.base_artifact_identity_digest and
  .adapter_artifact.serving_binding.residency_mode == "hot_swap_overlay" and
  (.adapter_artifact.serving_binding.served_adapter_digest | test("^[0-9a-f]{64}$")) and
  .adapter_artifact.loaded_rank == 2 and
  .adapter_artifact.loaded_hidden_size == 4 and
  .adapter_artifact.loaded_vocabulary_size == 3 and
  .adapter_artifact.save_load_exact_match == true and
  .adapter_artifact.serve_logits_exact_match == true and
  .adapter_artifact.expected_logits_sha256 == .adapter_artifact.served_logits_sha256 and
  .adapter_artifact.deterministic_serve_replay == true and
  .adapter_artifact.tampered_artifact_refused == true and
  .adapter_artifact.metadata_drift_refused == true and
  (.adapter_artifact.artifact_sha256 | test("^[0-9a-f]{64}$")) and
  (.adapter_artifact.receipt_digest | test("^[0-9a-f]{64}$")) and
  .refusals.inherited_model.refusal.code == "unsupported_model" and
  .refusals.inherited_adapter.refusal.code == "inherited_adapter_identity" and
  .refusals.decoder_target.refusal.code == "unsupported_target" and
  .refusals.native_cuda.refusal.code == "unsupported_execution_mode" and
  .refusals.artifact_drift.refusal.code == "base_artifact_identity_mismatch" and
  .refusals.missing_lineage.refusal.code == "missing_lineage" and
  ([.refusals[].status] | all(. == "refused")) and
  .real_checkpoint_training_admitted == false and
  .native_backward_admitted == false and
  .adapter_artifact_written == true and
  .tiny_reference_adapter_serving_parity_admitted == true and
  .adapter_serving_admitted == false and
  .tiny_reference_checkpoint_recovery_admitted == true and
  .checkpoint_recovery_admitted == false and
  .promotion_admitted == false and
  (.report_digest | test("^[0-9a-f]{64}$"))
' "${REPORT_PATH}" >/dev/null

readonly EXPECTED_ARTIFACT_SHA256="$(jq -r '.adapter_artifact.artifact_sha256' "${REPORT_PATH}")"
readonly ACTUAL_ARTIFACT_SHA256="$(sha256sum "${ARTIFACT_PATH}" | awk '{print $1}')"
readonly EXPECTED_ARTIFACT_BYTES="$(jq -r '.adapter_artifact.artifact_byte_length' "${REPORT_PATH}")"
readonly ACTUAL_ARTIFACT_BYTES="$(stat -c '%s' "${ARTIFACT_PATH}")"

[[ "${ACTUAL_ARTIFACT_SHA256}" == "${EXPECTED_ARTIFACT_SHA256}" ]]
[[ "${ACTUAL_ARTIFACT_BYTES}" == "${EXPECTED_ARTIFACT_BYTES}" ]]

echo "qwen38 training/adapter evidence passed: ${REPORT_PATH} ${ARTIFACT_PATH}"

#!/usr/bin/env bash
set -euo pipefail

readonly REPORT="${1:-fixtures/qwen38/reports/qwen38_cpu_mtp_evidence_v1.json}"
readonly EXPECTED_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"

jq -e \
  --arg artifact_sha256 "$EXPECTED_SHA256" \
  '
    .schema_version == "psionic.qwen38.mtp_real_artifact_evidence.v1"
    and (.source.revision | test("^[0-9a-f]{40}$"))
    and .source.dirty == false
    and .artifact.byte_length == 13441059904
    and .artifact.sha256 == $artifact_sha256
    and .backend == "native_psionic_cpu"
    and .prompt_token_ids == [9419, 11]
    and .max_output_tokens == 2
    and .correctness.output_token_parity == true
    and .correctness.output_text_parity == true
    and .correctness.restored_state_parity == true
    and .correctness.passed == true
    and .mtp_execution.schema_version == "psionic.qwen38.mtp_execution.v1"
    and .mtp_execution.backend == "cpu"
    and .mtp_execution.enabled == true
    and .mtp_execution.max_draft_tokens_per_cycle == 1
    and .mtp_execution.draft_count > 0
    and (.mtp_execution.accepted_count + .mtp_execution.rejected_count)
      == .mtp_execution.draft_count
    and .mtp_execution.mtp_forward_count
      == (.mtp_execution.draft_count + .mtp_execution.mtp_alignment_forward_count)
    and .mtp_execution.mtp_alignment_forward_count == .mtp_execution.accepted_count
    and .mtp_execution.rollback_count == .mtp_execution.rejected_count
    and .mtp_execution.target_replay_count == .mtp_execution.rejected_count
    and .mtp_execution.restored_state_parity == true
    and .mtp_execution.mtp_weight_residency_bytes == 208427008
    and .mtp_execution.mtp_kv_cache_peak_bytes > 0
    and .mtp_execution.rollback_snapshot_peak_bytes > 0
    and .mtp_execution.added_peak_residency_bytes
      == (.mtp_execution.mtp_weight_residency_bytes
        + .mtp_execution.mtp_kv_cache_peak_bytes
        + .mtp_execution.rollback_snapshot_peak_bytes)
    and .mtp_execution.performance_claim == "correctness_only_no_acceleration_claim"
    and .baseline.decode_tokens_per_second > 0
    and .mtp.decode_tokens_per_second > 0
    and .performance.baseline_decode_tokens_per_second
      == .baseline.decode_tokens_per_second
    and .performance.mtp_decode_tokens_per_second
      == .mtp.decode_tokens_per_second
    and .performance.acceleration_claimed == false
    and (.performance.observed_outcome
      | IN("slowdown_observed", "single_run_speedup_observed_not_claimed"))
    and .all_passed == true
  ' \
  "$REPORT" >/dev/null

echo "qwen38 CPU MTP evidence passed: $REPORT"

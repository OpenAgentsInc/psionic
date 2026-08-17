#!/usr/bin/env bash
set -euo pipefail

readonly REPORT="${1:-fixtures/qwen38/reports/qwen38_metal_generation_evidence_v1.json}"
readonly EXPECTED_SHA256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

jq -e \
  --arg artifact_sha256 "$EXPECTED_SHA256" \
  '
    .schema_version == "psionic.qwen38.metal_generation_evidence.v1"
    and (.source.revision | test("^[0-9a-f]{40}$"))
    and .source.dirty == false
    and .artifact.byte_length == 17106775008
    and .artifact.sha256 == $artifact_sha256
    and .hardware.memory_bytes > 0
    and .hardware.architecture == "arm64"
    and .serial_execution.variants == ["cpu", "metal"]
    and .serial_execution.parallel == false
    and .serial_execution.competing_processes == []
    and .correctness.prompt_token_parity == true
    and .correctness.cpu_metal_token_parity == true
    and .correctness.metal_reset_token_parity == true
    and .correctness.metal_reset_text_parity == true
    and .residency.family == "qwen38"
    and .residency.execution_plan_namespace == "qwen38-native-metal|v1"
    and .residency.context_limit_tokens == 4096
    and .residency.artifact_bytes == 17106775008
    and .residency.weight_device_bytes > 0
    and .residency.recurrent_state_host_bytes > 0
    and .residency.kv_cache_host_bytes > 0
    and .residency.admitted_layer_count == .residency.resident_layer_count
    and .residency.projection_count == .residency.native_projection_count
    and .residency.admitted_conversion_count == 0
    and .residency.host_stepped_state == true
    and .residency.host_projection_fallback_enabled == false
    and .publication.backend == "metal"
    and .publication.execution_mode == "native"
    and .publication.execution_engine == "psionic"
    and .publication.fallback_policy == "refuse"
    and .refusal.unsupported_projection_policy == "refuse_before_execution"
    and .cpu.backend == "cpu"
    and .metal.backend == "metal"
    and .cpu.mean_decode_tok_s > 0
    and .metal.mean_decode_tok_s > 0
    and .all_passed == true
  ' \
  "$REPORT" >/dev/null

echo "qwen38 Metal generation evidence passed: $REPORT"

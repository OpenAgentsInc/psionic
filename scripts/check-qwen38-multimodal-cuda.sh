#!/usr/bin/env bash
set -euo pipefail

readonly REPORT_PATH="${1:-fixtures/qwen38/reports/qwen38_multimodal_cuda_evidence_v1.json}"
readonly EXPECTED_SOURCE_SHARD_SHA256="ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
readonly EXPECTED_GGUF_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly EXPECTED_IMAGE_PARITY_SHA256="cccf9763ffc3ceaf78c5e2ec532f8112ee8fbc71efdde01f838ab76562562217"
readonly EXPECTED_VIDEO_PARITY_SHA256="b5e1908dd0e0f0180b145ae74878643ea61e5e592e62010174d440d339021d4f"

jq -e \
  --arg source_sha256 "${EXPECTED_SOURCE_SHARD_SHA256}" \
  --arg gguf_sha256 "${EXPECTED_GGUF_SHA256}" \
  --arg image_parity_sha256 "${EXPECTED_IMAGE_PARITY_SHA256}" \
  --arg video_parity_sha256 "${EXPECTED_VIDEO_PARITY_SHA256}" \
  '
    .schema_version == "psionic.qwen38.multimodal_cuda_evidence.v1" and
    .status == "implemented_early" and
    .phase == "R11" and
    (.psionic_revision | test("^[0-9a-f]{40}$")) and
    .artifacts.official_model_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" and
    .artifacts.vision_source_shard.byte_length == 3966730552 and
    .artifacts.vision_source_shard.sha256 == $source_sha256 and
    .artifacts.decoder_gguf.byte_length == 13441059904 and
    .artifacts.decoder_gguf.sha256 == $gguf_sha256 and
    .linked_encoder_parity.image_report_sha256 == $image_parity_sha256 and
    .linked_encoder_parity.video_report_sha256 == $video_parity_sha256 and
    (.gpu.name | length) > 0 and
    (.gpu.driver_version | length) > 0 and
    (.rows | length) == 2 and
    ([.rows[].media_kind] | sort) == ["image", "video"] and
    ([.rows[].gpu_idle_check.idle] | all) and
    ([.rows[].gpu_idle_check.output] | all(. == "")) and
    ([.rows[].schema_version] | unique) == ["psionic.qwen38.multimodal_cuda_smoke.v1"] and
    ([.rows[].fallback_policy] | unique) == ["refuse"] and
    ([.rows[].hidden_fallback_used] | any) == false and
    ([.rows[].vision_runtime.schema_version] | unique) == ["psionic.qwen38.vision_runtime.v1"] and
    ([.rows[].vision_runtime.backend] | unique) == ["cuda"] and
    ([.rows[].vision_runtime.execution_mode] | unique) == ["native"] and
    ([.rows[].vision_runtime.execution_engine] | unique) == ["psionic_candle_qwen38_vision_cuda"] and
    ([.rows[].vision_runtime.fallback_policy] | unique) == ["refuse"] and
    ([.rows[].vision_runtime.source_shard_sha256] | unique) == [$source_sha256] and
    ([.rows[].vision_runtime.resident_tensor_count] | unique) == [333] and
    ([.rows[].vision_runtime.resident_tensor_bytes] | unique) == [921460192] and
    ([.rows[].vision_runtime.resident_layer_count] | unique) == [27] and
    ([.rows[].vision_runtime.full_stack_resident] | all) and
    ([.rows[].vision_runtime.output_width] | unique) == [5120] and
    ([.rows[].vision_runtime.hidden_fallback_used] | any) == false and
    ([.rows[].multimodal_plan_receipt.schema_version] | unique) == ["psionic.qwen38.multimodal_decoder_plan.v1"] and
    ([.rows[].multimodal_plan_receipt.embedding_width] | unique) == [5120] and
    ([.rows[].multimodal_plan_receipt.hidden_fallback_used] | any) == false and
    ([.rows[] | (.vision_runtime.output_sha256 == .multimodal_plan_receipt.vision_runtime_output_sha256[0])] | all) and
    ([.rows[].decoder_runtime.family] | unique) == ["qwen38"] and
    ([.rows[].decoder_runtime.context_limit_tokens] | unique) == [4096] and
    ([.rows[].decoder_runtime.artifact_bytes] | unique) == [13441059904] and
    ([.rows[].decoder_runtime.preflight_status] | unique) == ["admitted_before_weight_upload"] and
    ([.rows[].decoder_runtime.host_fallback_enabled] | any) == false and
    ([.rows[].decoder_runtime.dense_f16_mirror_count] | unique) == [0] and
    ([.rows[].generation_metrics.qwen35_cuda_decode.output_modes] | unique) == [[{"kind":"argmax_only"}]] and
    ([.rows[].generation_metrics.qwen35_cuda_decode.raw_logits_materialized] | any) == false and
    ([.rows[].generation_metrics.qwen35_cuda_decode.readback_bytes] | unique) == [16] and
    ([.rows[].generation_metrics.eval_count] | unique) == [2] and
    ([.rows[].termination] | unique) == ["max_output_tokens"] and
    ([.rows[].total_duration_ns] | min) > 0 and
    .claim_boundary.cuda_image_generation == true and
    .claim_boundary.cuda_video_generation == true and
    .claim_boundary.metal_decoder_integration == false and
    .claim_boundary.openai_media_serving == false and
    .claim_boundary.performance_claim == false
  ' "${REPORT_PATH}" >/dev/null

jq -e '
  (.rows[] | select(.media_kind == "image")) as $image |
  ($image.multimodal_plan_receipt.token_count == 87) and
  ($image.multimodal_plan_receipt.text_token_count == 23) and
  ($image.multimodal_plan_receipt.image_count == 1) and
  ($image.multimodal_plan_receipt.image_token_count == 64) and
  ($image.multimodal_plan_receipt.video_count == 0) and
  ($image.multimodal_plan_receipt.embedding_override_count == 64) and
  ($image.vision_runtime.output_token_count == 64) and
  ($image.output_token_ids == [760, 2099]) and
  ($image.output_text == "The image") and
  ($image.generation_metrics.prompt_eval_count == 87)
' "${REPORT_PATH}" >/dev/null

jq -e '
  (.rows[] | select(.media_kind == "video")) as $video |
  ($video.multimodal_plan_receipt.token_count == 167) and
  ($video.multimodal_plan_receipt.text_token_count == 39) and
  ($video.multimodal_plan_receipt.image_count == 0) and
  ($video.multimodal_plan_receipt.video_count == 1) and
  ($video.multimodal_plan_receipt.video_token_count == 128) and
  ($video.multimodal_plan_receipt.embedding_override_count == 128) and
  ($video.vision_runtime.output_token_count == 128) and
  ($video.output_token_ids == [760, 2678]) and
  ($video.output_text == "The video") and
  ($video.generation_metrics.prompt_eval_count == 167)
' "${REPORT_PATH}" >/dev/null

echo "qwen38 multimodal CUDA evidence passed: ${REPORT_PATH}"

#!/usr/bin/env bash
set -euo pipefail

readonly REPORT_PATH="${1:-fixtures/qwen38/reports/qwen38_openai_media_evidence_v1.json}"
readonly EXPECTED_SOURCE_SHARD_SHA256="ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
readonly EXPECTED_GGUF_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"

jq -e \
  --arg source_sha256 "${EXPECTED_SOURCE_SHARD_SHA256}" \
  --arg gguf_sha256 "${EXPECTED_GGUF_SHA256}" \
  '
    .schema_version == "psionic.qwen38.openai_media_evidence.v1" and
    .status == "implemented_early" and
    .phase == "R11" and
    (.psionic_revision | test("^[0-9a-f]{40}$")) and
    .artifacts.official_model_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" and
    .artifacts.vision_source_shard.byte_length == 3966730552 and
    .artifacts.vision_source_shard.sha256 == $source_sha256 and
    .artifacts.decoder_gguf.byte_length == 13441059904 and
    .artifacts.decoder_gguf.sha256 == $gguf_sha256 and
    .gpu.idle_check.idle == true and
    .gpu.idle_check.output == "" and
    (.gpu.name | length) > 0 and
    (.gpu.driver_version | length) > 0 and
    (.server_log_sha256 | test("^[0-9a-f]{64}$")) and
    .server.health.status == 200 and
    .server.health.body.status == "ok" and
    .server.health.body.backend == "cuda" and
    .server.health.body.execution_mode == "native" and
    .server.health.body.execution_engine == "psionic" and
    .server.health.body.qwen38.vision_backend == "cpu" and
    .server.health.body.qwen38.image_input == "native_bounded_base64_data_url" and
    .server.health.body.qwen38.video_input == "native_bounded_animated_gif_data_url" and
    ([.server.models.body.data[] | select(.psionic_model_family == "qwen38")] | length) == 1 and
    .request_contract.video_source_frames == 8 and
    .request_contract.video_source_fps == 4 and
    .request_contract.chat_tools_present == true and
    .request_contract.responses_official_input_parts == ["input_image", "input_text"] and
    .claim_boundary.cpu_vision == true and
    .claim_boundary.cuda_decoder == true and
    .claim_boundary.chat_image_generation == true and
    .claim_boundary.chat_video_streaming == true and
    .claim_boundary.responses_image_generation == true and
    .claim_boundary.metal_decoder_integration == false and
    .claim_boundary.remote_media_fetch == false and
    .claim_boundary.mp4_decode == false and
    .claim_boundary.responses_binary_media_replay == false and
    .claim_boundary.performance_claim == false
  ' "${REPORT_PATH}" >/dev/null

jq -e '
  .generation.chat_image_with_tools as $row |
  $row.status == 200 and
  ($row.body.choices[0].message.content | length) > 0 and
  $row.body.psionic_qwen38.backend == "cuda" and
  $row.body.psionic_qwen38.execution_mode == "native" and
  $row.body.psionic_qwen38.execution_engine == "psionic" and
  $row.body.psionic_qwen38.prompt.tool_count == 1 and
  $row.body.psionic_qwen38.multimodal.input_contract == "bounded_base64_data_urls" and
  $row.body.psionic_qwen38.multimodal.vision_backend == "cpu" and
  $row.body.psionic_qwen38.multimodal.vision_execution_engine == "psionic_candle_qwen38_vision_cpu" and
  $row.body.psionic_qwen38.multimodal.attachment_count == 1 and
  $row.body.psionic_qwen38.multimodal.attachments[0].media_kind == "image" and
  $row.body.psionic_qwen38.multimodal.attachments[0].source_transport == "data_url_base64" and
  ($row.body.psionic_qwen38.multimodal.attachments[0].source_sha256 | test("^[0-9a-f]{64}$")) and
  $row.body.psionic_qwen38.multimodal.preprocessing[0].media_kind == "image" and
  $row.body.psionic_qwen38.multimodal.preprocessing[0].width == 256 and
  $row.body.psionic_qwen38.multimodal.preprocessing[0].height == 256 and
  $row.body.psionic_qwen38.multimodal.vision_runtime[0].backend == "cpu" and
  $row.body.psionic_qwen38.multimodal.vision_runtime[0].fallback_policy == "refuse" and
  $row.body.psionic_qwen38.multimodal.decoder_plan.image_count == 1 and
  $row.body.psionic_qwen38.multimodal.decoder_plan.video_count == 0 and
  $row.body.psionic_qwen38.multimodal.hidden_fallback_used == false
' "${REPORT_PATH}" >/dev/null

jq -e '
  .generation.chat_video_stream as $row |
  $row.status == 200 and
  $row.event_count >= 2 and
  ($row.output_text | length) > 0 and
  $row.headers["x-psionic-backend"] == "cuda" and
  $row.headers["x-psionic-qwen38-media-attachments"] == "1" and
  $row.headers["x-psionic-qwen38-vision-backend"] == "cpu" and
  ($row.headers["x-psionic-qwen38-attachment-sha256"] | test("^[0-9a-f]{64}$")) and
  ($row.headers["x-psionic-qwen38-multimodal-token-sha256"] | test("^[0-9a-f]{64}$")) and
  ($row.headers["x-psionic-qwen38-expanded-prompt-sha256"] | test("^[0-9a-f]{64}$"))
' "${REPORT_PATH}" >/dev/null

jq -e '
  .generation.responses_image as $row |
  $row.status == 200 and
  ($row.body.output_text | length) > 0 and
  $row.body.psionic_response_state.stored == true and
  $row.body.psionic_qwen38.backend == "cuda" and
  $row.body.psionic_qwen38.multimodal.attachment_count == 1 and
  $row.body.psionic_qwen38.multimodal.attachments[0].media_kind == "image" and
  $row.body.psionic_qwen38.multimodal.decoder_plan.image_count == 1 and
  $row.body.psionic_qwen38.multimodal.decoder_plan.video_count == 0 and
  $row.body.psionic_qwen38.multimodal.hidden_fallback_used == false
' "${REPORT_PATH}" >/dev/null

jq -e '
  .refusals.responses_media_continuation as $continuation |
  .refusals.remote_image as $remote |
  .refusals.mp4_video as $mp4 |
  .refusals.malformed_base64 as $malformed |
  .refusals.five_attachments as $excess |
  $continuation.status == 400 and
  ($continuation.body.error.message | test("cannot replay retained binary media")) and
  $remote.status == 400 and
  ($remote.body.error.message | test("remote URLs are refused")) and
  $mp4.status == 400 and
  ($mp4.body.error.message | test("animated image/gif")) and
  $malformed.status == 400 and
  ($malformed.body.error.message | test("invalid base64")) and
  $excess.status == 400 and
  ($excess.body.error.message | test("at most 4 attachments"))
' "${REPORT_PATH}" >/dev/null

echo "qwen38 OpenAI media evidence passed: ${REPORT_PATH}"

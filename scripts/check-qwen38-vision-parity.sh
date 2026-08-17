#!/usr/bin/env bash
set -euo pipefail

REPORT_PATH="${1:-fixtures/qwen38/reports/qwen38_vision_parity_v1.json}"

jq -e '
  .schema_version == "psionic.qwen38.vision_parity.v1" and
  .status == "implemented_early" and
  .native.preprocessing.schema_version == "psionic.qwen38.vision_preprocessing.v1" and
  .native.runtime.schema_version == "psionic.qwen38.vision_runtime.v1" and
  .native.runtime.backend == "cuda" and
  .native.runtime.execution_mode == "native" and
  .native.runtime.fallback_policy == "refuse" and
  .native.runtime.resident_tensor_count == 333 and
  .native.runtime.resident_tensor_bytes == 921460192 and
  .native.runtime.resident_layer_count == 27 and
  .native.runtime.full_stack_resident == true and
  .native.runtime.hidden_fallback_used == false and
  .reference.transformers_revision == "0650ff354501cbdb7cb4138da628cc60f4e0ceed" and
  .reference.transformers_version == "5.16.0.dev0" and
  .reference.torch_version == "2.11.0+cu128" and
  .reference.source_shard_sha256 == "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c" and
  .reference.processor_backend == "torchvision" and
  .native.preprocessing.processor_name == .reference.processor_class and
  .native.preprocessing.media_kind == .reference.media_kind and
  .sampling_parity.passed == true and
  .processor_parity.passed == true and
  .output_parity.passed == true and
  .claim_boundary.text_decoder_integration == false and
  .claim_boundary.openai_serving == false and
  (
    (
      .claim_boundary.media_kind == "image" and
      .claim_boundary.image_encoder_parity == true and
      .claim_boundary.video_encoder_parity == false and
      .reference.processor_class == "Qwen2VLImageProcessor"
    ) or
    (
      .claim_boundary.media_kind == "video" and
      .claim_boundary.image_encoder_parity == false and
      .claim_boundary.video_encoder_parity == true and
      .reference.processor_class == "Qwen3VLVideoProcessor"
    )
  )
' "$REPORT_PATH" >/dev/null

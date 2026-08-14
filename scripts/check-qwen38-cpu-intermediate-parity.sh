#!/usr/bin/env bash
set -euo pipefail

readonly REPORT="${1:-fixtures/qwen38/reports/qwen38_cpu_recurrent_intermediate_parity_v1.json}"
readonly EXPECTED_ARTIFACT_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly EXPECTED_LLAMA_CPP_REVISION="9b05354ec6fb58b4e665e9a39ebc40285c015638"

jq -e \
  --arg artifact_sha256 "$EXPECTED_ARTIFACT_SHA256" \
  --arg llama_revision "$EXPECTED_LLAMA_CPP_REVISION" \
  '
    .schema_version == "psionic_qwen38_cpu_recurrent_intermediate_parity_v1"
    and .artifact.sha256 == $artifact_sha256
    and .comparator.revision == $llama_revision
    and .comparator.trace_schema_version == "qwen38_llama_cpp_recurrent_trace_v1"
    and .comparator.backend == "cpu"
    and .psionic.implementation == "native_psionic_cpu"
    and .psionic.backend == "cpu"
    and .tokens.prefill_token_ids == [9419, 11]
    and .tokens.retained_decode_token_ids == [353]
    and .all_passed == true
    and (.comparisons | length) == 28
    and ([.comparisons[].phase] | map(select(. == "prefill")) | length) == 14
    and ([.comparisons[].phase] | map(select(. == "decode")) | length) == 14
    and ([.comparisons[].stage] | unique | length) == 14
    and ([.comparisons[].passed] | all)
    and ([.comparisons[].metrics.normalized_rmse] | max) <= 0.012
    and ([.comparisons[].metrics.cosine_similarity] | min) >= 0.9999
    and ([.comparisons[] | select(.stage == "new_state") | .state_layout] | unique)
      == ["ggml_transposed_state_direct"]
  ' \
  "$REPORT" >/dev/null

echo "qwen38 CPU recurrent-intermediate parity report passed: $REPORT"

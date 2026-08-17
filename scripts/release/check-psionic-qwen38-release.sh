#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPORT="${PSIONIC_QWEN38_RELEASE_REPORT:-${REPO_ROOT}/fixtures/qwen38/reports/qwen38_release_gate_v1.json}"
readonly EXPECTED_ARTIFACT_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly EXPECTED_LLAMA_CPP_REVISION="9b05354ec6fb58b4e665e9a39ebc40285c015638"

cd "${REPO_ROOT}"

if [[ "${PSIONIC_QWEN38_RELEASE_ALLOW_DIRTY:-0}" != "1" ]] &&
  [[ -n "$(git status --porcelain)" ]]; then
  echo "Qwen3.8 release checking requires a clean Psionic checkout" >&2
  exit 1
fi

if [[ ! -f "${REPORT}" ]]; then
  echo "missing Qwen3.8 release report: ${REPORT}" >&2
  exit 1
fi

if grep -Eqi 'qwen35|qwen3[._ -]?5' "${REPORT}"; then
  echo "Qwen3.8 release report contains a Qwen3.5 name or schema" >&2
  exit 1
fi

report_revision="$(jq -er '.psionic.revision' "${REPORT}")"
git cat-file -e "${report_revision}^{commit}"
git merge-base --is-ancestor "${report_revision}" HEAD

check_digest() {
  local evidence_id="$1"
  local path expected actual
  path="$(jq -er --arg evidence_id "${evidence_id}" \
    '.evidence[] | select(.evidence_id == $evidence_id) | .path' "${REPORT}")"
  expected="$(jq -er --arg evidence_id "${evidence_id}" \
    '.evidence[] | select(.evidence_id == $evidence_id) | .sha256' "${REPORT}")"
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Qwen3.8 release evidence digest mismatch for ${evidence_id}: ${path}" >&2
    exit 1
  fi
}

for evidence_id in \
  artifact_facts \
  prompt_tokenizer_golden \
  release_template_cases \
  cpu_recurrent_parity \
  cuda_greedy_generation \
  cuda_bounded_sample_generation; do
  check_digest "${evidence_id}"
done

jq -e \
  --arg artifact_sha256 "${EXPECTED_ARTIFACT_SHA256}" \
  --arg llama_revision "${EXPECTED_LLAMA_CPP_REVISION}" \
  '
    .schema_version == "psionic.qwen38.release_gate.v1"
    and .report_kind == "qwen38_release_gate"
    and .issue == 1151
    and .status == "passed"
    and .all_passed == true
    and .generated_from_clean_checkout == true
    and .psionic.branch == "main"
    and .psionic.dirty == false
    and .psionic.upstream == "origin/main"
    and (.psionic.revision | length) == 40
    and .model.product_family == "qwen38"
    and .model.official_model_id == "Qwen/Qwen3.8-27B"
    and .model.served_model_id == "qwen3.8-27b"
    and .model.upstream_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    and .model.config_sha256 == "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
    and .model.template_sha256 == "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    and .model.tokenizer_sha256 == "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
    and .artifact.filename == "Qwen3.8-27B-UD-Q3_K_XL.gguf"
    and .artifact.byte_length == 13441059904
    and .artifact.sha256 == $artifact_sha256
    and .runtime.backends == ["cpu", "cuda"]
    and .runtime.context_tokens == 4096
    and .runtime.host_fallback_enabled == false
    and .runtime.raw_logits_observable == true
    and .runtime.generic_server_status == "implemented_early"
    and .prompt_contract.prompt_text == "Hello"
    and .prompt_contract.prompt_token_ids == [9419]
    and .prompt_contract.expected_greedy_output_token_ids == [11, 353]
    and .prompt_contract.expected_seeded_sample_output_token_ids == [11, 271]
    and .prompt_contract.sample_settings == {
      "seed": 42,
      "temperature": 0.8,
      "top_k": 40,
      "top_p": 0.9
    }
    and .correctness.direct_native_generation == "passed"
    and .correctness.generic_server_generation == "passed"
    and .correctness.tool_loop_replay == "passed"
    and .correctness.structured_output_acceptance_and_refusal == "passed"
    and .correctness.repeated_request_state_reset == "passed"
    and .comparator.implementation == "ggml-org/llama.cpp"
    and .comparator.revision == $llama_revision
    and .comparator.backend == "cpu"
    and .comparator.generation_intermediate_comparisons == 28
    and .comparator.generation_intermediate_all_passed == true
    and .comparator.template.endpoint == "/apply-template"
    and .comparator.template.request_argument == "chat_template_kwargs"
    and .comparator.template.all_passed == true
    and (.comparator.template.cases | length) == 3
    and ([.comparator.template.cases[].reasoning_effort] | sort) == ["low", "medium", "xhigh"]
    and ([.comparator.template.cases[].passed] | all)
    and ([.comparator.template.cases[] | .actual_rendered_sha256 == .expected_rendered_sha256] | all)
    and .cuda_publication.host_fallback_invocations == 0
    and .cuda_publication.graph_hits > 0
    and .cuda_publication.graph_shape_drifts == 0
    and (.cuda_publication.graph_cache_identity | length) == 64
    and .cuda_publication.allocator_peak_resident_device_bytes == 13390641048
    and .cuda_publication.allocator_peak_resident_device_bytes
      <= .cuda_publication.preflight_required_device_bytes
    and .cuda_publication.context_tokens == 4096
    and .performance.claim_status == "not_published"
    and .performance.release_bar == "correctness_and_truthful_runtime_publication"
    and .performance.observations.cuda_greedy_mean_decode_tok_s > 0
    and .performance.observations.cuda_sample_mean_decode_tok_s > 0
    and .performance.observations.template_latency_ms_low >= 0
    and .performance.observations.template_latency_ms_medium >= 0
    and .performance.observations.template_latency_ms_xhigh >= 0
    and .refusals.validated_refusal_case_count >= 5
    and ([.refusals.capabilities[].feature] | sort) == [
      "adapters",
      "media_execution",
      "metal_backend",
      "mtp_speculative_decoding",
      "session_reuse",
      "training",
      "yarn_context_extension"
    ]
    and ([.refusals.capabilities[].status] | all(. == "refused" or . == "planned"))
    and ([.validation.gates[].gate_id] | sort) == [
      "artifact_contract",
      "cpu_recurrent_comparator",
      "cuda_publication",
      "direct_native_generation",
      "openai_serving",
      "prompt_contract",
      "qwen36_regression"
    ]
    and ([.validation.gates[].passed] | all)
    and ([.evidence[].schema_version] | all(
      if type == "number" then
        . == 7
      else
        startswith("psionic.qwen38") or startswith("psionic_qwen38")
      end
    ))
  ' "${REPORT}" >/dev/null

scripts/check-qwen38-cpu-intermediate-parity.sh
scripts/check-qwen38-cuda-generation.sh

if [[ "${PSIONIC_QWEN38_RELEASE_SKIP_TESTS:-0}" != "1" ]]; then
  cargo test -p psionic-models qwen38_artifact -- --test-threads=1
  cargo test -p psionic-models qwen38_prompt -- --test-threads=1
  cargo test -p psionic-models qwen36 -- --test-threads=1
  cargo test -p psionic-serve --lib qwen38_cpu_generation_skips_mtp_and_resets_request_state -- --test-threads=1
  cargo test -p psionic-serve --lib qwen38_openai -- --test-threads=1
fi

echo "Qwen3.8 release gate passed: ${REPORT}"

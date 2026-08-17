#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPORT="${PSIONIC_QWEN38_RELEASE_REPORT:-${REPO_ROOT}/fixtures/qwen38/reports/qwen38_release_gate_v1.json}"
readonly ARTIFACT_FACTS="fixtures/qwen38/qwen38_27b_artifact_facts_v1.json"
readonly PROMPT_GOLDEN="fixtures/qwen38/qwen38_prompt_tokenizer_golden_v1.json"
readonly TEMPLATE_CASES="fixtures/qwen38/qwen38_release_template_cases_v1.json"
readonly CPU_REPORT="fixtures/qwen38/reports/qwen38_cpu_recurrent_intermediate_parity_v1.json"
readonly CUDA_GREEDY_REPORT="fixtures/qwen38/reports/qwen38_cuda_greedy_generation_v1.json"
readonly CUDA_SAMPLE_REPORT="fixtures/qwen38/reports/qwen38_cuda_bounded_sample_generation_v1.json"
readonly ARTIFACT="${PSIONIC_QWEN38_RELEASE_GGUF:-${REPO_ROOT}/target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
readonly LLAMA_CPP_ROOT="${PSIONIC_QWEN38_LLAMA_CPP_ROOT:-/home/christopherdavid/code/llama.cpp}"
readonly LLAMA_SERVER="${PSIONIC_QWEN38_LLAMA_SERVER:-${LLAMA_CPP_ROOT}/build-cpu/bin/llama-server}"
readonly LLAMA_REVISION="9b05354ec6fb58b4e665e9a39ebc40285c015638"
readonly ARTIFACT_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly ARTIFACT_BYTES=13441059904
readonly PORT="${PSIONIC_QWEN38_LLAMA_PORT:-18089}"

cd "${REPO_ROOT}"

for tool in cargo curl git jq sha256sum; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "missing required Qwen3.8 release tool: ${tool}" >&2
    exit 1
  fi
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Qwen3.8 release evidence must start from a clean Psionic checkout" >&2
  exit 1
fi

branch="$(git branch --show-current)"
upstream="$(git rev-parse --abbrev-ref '@{upstream}')"
revision="$(git rev-parse HEAD)"
if [[ "${branch}" != "main" || "${upstream}" != "origin/main" ]]; then
  echo "Qwen3.8 release evidence requires main tracking origin/main" >&2
  exit 1
fi
if [[ "${revision}" != "$(git rev-parse origin/main)" ]]; then
  echo "Qwen3.8 release evidence requires main to match origin/main" >&2
  exit 1
fi

for path in \
  "${ARTIFACT_FACTS}" \
  "${PROMPT_GOLDEN}" \
  "${TEMPLATE_CASES}" \
  "${CPU_REPORT}" \
  "${CUDA_GREEDY_REPORT}" \
  "${CUDA_SAMPLE_REPORT}" \
  "${ARTIFACT}" \
  "${LLAMA_SERVER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing Qwen3.8 release input: ${path}" >&2
    exit 1
  fi
done

actual_artifact_bytes="$(stat -c '%s' "${ARTIFACT}")"
actual_artifact_sha256="$(sha256sum "${ARTIFACT}" | cut -d' ' -f1)"
if [[ "${actual_artifact_bytes}" != "${ARTIFACT_BYTES}" ||
  "${actual_artifact_sha256}" != "${ARTIFACT_SHA256}" ]]; then
  echo "Qwen3.8 release artifact does not match the pinned byte length and digest" >&2
  exit 1
fi

llama_revision="$(git -C "${LLAMA_CPP_ROOT}" rev-parse HEAD)"
if [[ "${llama_revision}" != "${LLAMA_REVISION}" ]]; then
  echo "llama.cpp revision mismatch: expected ${LLAMA_REVISION}, got ${llama_revision}" >&2
  exit 1
fi
if [[ -n "$(git -C "${LLAMA_CPP_ROOT}" status --porcelain)" ]]; then
  echo "Qwen3.8 release comparator requires a clean llama.cpp checkout" >&2
  exit 1
fi
llama_version="$(${LLAMA_SERVER} --version 2>&1)"
if [[ "${llama_version}" != *"commit ${LLAMA_REVISION:0:9}"* ]]; then
  echo "llama.cpp server binary was not built from ${LLAMA_REVISION}" >&2
  exit 1
fi

tmp_dir="$(mktemp -d "${REPO_ROOT}/target/qwen38-release.XXXXXX")"
gate_rows="${tmp_dir}/gate-rows.jsonl"
template_rows="${tmp_dir}/template-rows.jsonl"
server_log="${tmp_dir}/llama-server.log"
server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

run_gate() {
  local gate_id="$1"
  shift
  local started_ns finished_ns duration_ms command_text
  started_ns="$(date +%s%N)"
  "$@"
  finished_ns="$(date +%s%N)"
  duration_ms="$(((finished_ns - started_ns) / 1000000))"
  printf -v command_text '%q ' "$@"
  jq -cn \
    --arg gate_id "${gate_id}" \
    --arg command "${command_text% }" \
    --argjson duration_ms "${duration_ms}" \
    '{gate_id: $gate_id, command: $command, duration_ms: $duration_ms, passed: true}' \
    >>"${gate_rows}"
}

run_gate artifact_contract \
  cargo test -p psionic-models qwen38_artifact -- --test-threads=1
run_gate prompt_contract \
  cargo test -p psionic-models qwen38_prompt -- --test-threads=1
run_gate qwen36_regression \
  cargo test -p psionic-models qwen36 -- --test-threads=1
run_gate direct_native_generation \
  cargo test -p psionic-serve --lib qwen38_cpu_generation_skips_mtp_and_resets_request_state -- --test-threads=1
run_gate openai_serving \
  cargo test -p psionic-serve --lib qwen38_openai -- --test-threads=1
run_gate cpu_recurrent_comparator scripts/check-qwen38-cpu-intermediate-parity.sh
run_gate cuda_publication scripts/check-qwen38-cuda-generation.sh

if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Qwen3.8 llama.cpp comparator port is already in use: ${PORT}" >&2
  exit 1
fi

"${LLAMA_SERVER}" \
  --model "${ARTIFACT}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --ctx-size 512 \
  --parallel 1 \
  --gpu-layers 0 \
  --no-op-offload \
  --no-warmup \
  --threads 1 \
  --threads-batch 1 \
  --jinja \
  >"${server_log}" 2>&1 &
server_pid="$!"

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    cat "${server_log}" >&2
    echo "Qwen3.8 llama.cpp comparator exited during startup" >&2
    exit 1
  fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  cat "${server_log}" >&2
  echo "Qwen3.8 llama.cpp comparator failed its health check" >&2
  exit 1
fi

while IFS= read -r template_case; do
  case_id="$(jq -r '.case_id' <<<"${template_case}")"
  reasoning_effort="$(jq -r '.chat_template_kwargs.reasoning_effort' <<<"${template_case}")"
  expected_sha256="$(jq -r '.expected_rendered_sha256' <<<"${template_case}")"
  request="$(jq -c '{messages, chat_template_kwargs}' <<<"${template_case}")"
  response_path="${tmp_dir}/${case_id}.json"
  elapsed_s="$(curl -fsS \
    -H 'Content-Type: application/json' \
    --data-binary "${request}" \
    --output "${response_path}" \
    --write-out '%{time_total}' \
    "http://127.0.0.1:${PORT}/apply-template")"
  actual_sha256="$(jq -j '.prompt' "${response_path}" | sha256sum | cut -d' ' -f1)"
  rendered_bytes="$(jq -j '.prompt' "${response_path}" | wc -c)"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Qwen3.8 template comparator mismatch for ${case_id}" >&2
    exit 1
  fi
  jq -cn \
    --arg case_id "${case_id}" \
    --arg reasoning_effort "${reasoning_effort}" \
    --arg expected_sha256 "${expected_sha256}" \
    --arg actual_sha256 "${actual_sha256}" \
    --arg elapsed_s "${elapsed_s}" \
    --argjson rendered_bytes "${rendered_bytes}" \
    '{
      case_id: $case_id,
      reasoning_effort: $reasoning_effort,
      expected_rendered_sha256: $expected_sha256,
      actual_rendered_sha256: $actual_sha256,
      rendered_bytes: $rendered_bytes,
      latency_ms: (($elapsed_s | tonumber) * 1000),
      passed: true
    }' >>"${template_rows}"
done < <(jq -c '.cases[]' "${TEMPLATE_CASES}")

kill "${server_pid}"
wait "${server_pid}"
server_pid=""

gate_results="$(jq -s '.' "${gate_rows}")"
template_results="$(jq -s '.' "${template_rows}")"
generated_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
commit_timestamp="$(git show -s --format='%cI' "${revision}")"

artifact_facts_sha256="$(sha256sum "${ARTIFACT_FACTS}" | cut -d' ' -f1)"
prompt_golden_sha256="$(sha256sum "${PROMPT_GOLDEN}" | cut -d' ' -f1)"
template_cases_sha256="$(sha256sum "${TEMPLATE_CASES}" | cut -d' ' -f1)"
cpu_report_sha256="$(sha256sum "${CPU_REPORT}" | cut -d' ' -f1)"
cuda_greedy_sha256="$(sha256sum "${CUDA_GREEDY_REPORT}" | cut -d' ' -f1)"
cuda_sample_sha256="$(sha256sum "${CUDA_SAMPLE_REPORT}" | cut -d' ' -f1)"

mkdir -p "$(dirname "${REPORT}")"
jq -n \
  --arg generated_at "${generated_at}" \
  --arg revision "${revision}" \
  --arg branch "${branch}" \
  --arg upstream "${upstream}" \
  --arg commit_timestamp "${commit_timestamp}" \
  --arg artifact_path "${ARTIFACT}" \
  --arg artifact_sha256 "${ARTIFACT_SHA256}" \
  --arg llama_revision "${llama_revision}" \
  --arg llama_version "${llama_version}" \
  --arg artifact_facts_sha256 "${artifact_facts_sha256}" \
  --arg prompt_golden_sha256 "${prompt_golden_sha256}" \
  --arg template_cases_sha256 "${template_cases_sha256}" \
  --arg cpu_report_sha256 "${cpu_report_sha256}" \
  --arg cuda_greedy_sha256 "${cuda_greedy_sha256}" \
  --arg cuda_sample_sha256 "${cuda_sample_sha256}" \
  --argjson artifact_bytes "${ARTIFACT_BYTES}" \
  --argjson gates "${gate_results}" \
  --argjson templates "${template_results}" \
  --slurpfile facts "${ARTIFACT_FACTS}" \
  --slurpfile cpu "${CPU_REPORT}" \
  --slurpfile greedy "${CUDA_GREEDY_REPORT}" \
  --slurpfile sample "${CUDA_SAMPLE_REPORT}" \
  '
    def suffixed($object; $suffix):
      $object | to_entries[] | select(.key | endswith($suffix)) | .value;
    ($templates | map({key: .reasoning_effort, value: .latency_ms}) | from_entries) as $latencies
    | {
      schema_version: "psionic.qwen38.release_gate.v1",
      report_kind: "qwen38_release_gate",
      issue: 1151,
      generated_at: $generated_at,
      generated_from_clean_checkout: true,
      status: "passed",
      psionic: {
        revision: $revision,
        branch: $branch,
        upstream: $upstream,
        commit_timestamp: $commit_timestamp,
        dirty: false
      },
      model: {
        product_family: "qwen38",
        official_model_id: $facts[0].identity.official_model_id,
        served_model_id: $facts[0].identity.served_model_id,
        upstream_revision: $facts[0].upstream_revision,
        config_sha256: $facts[0].digests.config_sha256,
        template_sha256: $facts[0].digests.chat_template_sha256,
        tokenizer_sha256: $facts[0].digests.tokenizer_sha256
      },
      artifact: {
        filename: "Qwen3.8-27B-UD-Q3_K_XL.gguf",
        local_path: $artifact_path,
        byte_length: $artifact_bytes,
        sha256: $artifact_sha256
      },
      runtime: {
        implementation: "native_psionic",
        backends: ["cpu", "cuda"],
        context_tokens: $greedy[0].qwen38_evidence.validation_scope.context_tokens,
        host_fallback_enabled: $greedy[0].psionic_cuda_startup.runtime_contract.host_fallback_enabled,
        raw_logits_observable: $greedy[0].psionic_cuda_startup.runtime_contract.raw_logits_materialization_observable,
        generic_server_status: "implemented_early",
        graph_cache_namespace: $greedy[0].psionic_cuda_startup.runtime_contract.graph_cache_namespace,
        execution_plan_namespace: $greedy[0].psionic_cuda_startup.runtime_contract.execution_plan_namespace
      },
      prompt_contract: {
        prompt_text: $greedy[0].prompt,
        prompt_token_ids: $greedy[0].prompt_token_ids,
        expected_greedy_output_token_ids: $greedy[0].qwen38_evidence.reference.greedy_output_token_ids,
        expected_seeded_sample_output_token_ids: $sample[0].runs[0].output_token_ids,
        sample_settings: {
          seed: $sample[0].seed,
          temperature: $sample[0].temperature,
          top_k: $sample[0].top_k,
          top_p: $sample[0].top_p
        }
      },
      correctness: {
        direct_native_generation: "passed",
        generic_server_generation: "passed",
        tool_loop_replay: "passed",
        structured_output_acceptance_and_refusal: "passed",
        repeated_request_state_reset: "passed"
      },
      comparator: {
        implementation: "ggml-org/llama.cpp",
        revision: $llama_revision,
        binary_version: $llama_version,
        backend: "cpu",
        generation_intermediate_comparisons: ($cpu[0].comparisons | length),
        generation_intermediate_all_passed: $cpu[0].all_passed,
        template: {
          endpoint: "/apply-template",
          request_argument: "chat_template_kwargs",
          cases: $templates,
          all_passed: ($templates | all(.passed))
        }
      },
      cuda_publication: {
        host_fallback_invocations: ([
          $greedy[0].runs[]
          | suffixed(.; "_host_fallback_evidence")
          | .fallback_invocations
        ] | add),
        graph_hits: ([
          $greedy[0].runs[] | suffixed(.; "_graph_hits")
        ] | add),
        graph_shape_drifts: ([
          $greedy[0].runs[] | suffixed(.; "_graph_shape_drifts")
        ] | add),
        graph_cache_identity: ([
          $greedy[0].runs[] | suffixed(.; "_graph_cache_identity")
        ] | unique | first),
        allocator_peak_resident_device_bytes: $greedy[0].qwen38_evidence.residency_measurement.allocator_peak_resident_device_bytes,
        allocator_resident_device_bytes_after_measurements: $greedy[0].qwen38_evidence.residency_measurement.allocator_resident_device_bytes_after_measurements,
        preflight_required_device_bytes: $greedy[0].psionic_cuda_startup.runtime_contract.preflight_required_device_bytes,
        device_free_bytes_at_preflight: $greedy[0].psionic_cuda_startup.runtime_contract.device_free_bytes_at_preflight,
        context_tokens: $greedy[0].qwen38_evidence.validation_scope.context_tokens
      },
      performance: {
        claim_status: "not_published",
        release_bar: "correctness_and_truthful_runtime_publication",
        observations: {
          cuda_greedy_mean_decode_tok_s: $greedy[0].mean_decode_tok_s,
          cuda_sample_mean_decode_tok_s: $sample[0].mean_decode_tok_s,
          template_latency_ms_low: $latencies.low,
          template_latency_ms_medium: $latencies.medium,
          template_latency_ms_xhigh: $latencies.xhigh
        }
      },
      refusals: {
        validated_refusal_case_count: 5,
        capabilities: [
          {feature: "adapters", status: "refused"},
          {feature: "media_execution", status: "refused"},
          {feature: "metal_backend", status: "refused"},
          {feature: "mtp_speculative_decoding", status: "planned"},
          {feature: "session_reuse", status: "refused"},
          {feature: "training", status: "planned"},
          {feature: "yarn_context_extension", status: "planned"}
        ]
      },
      validation: {
        gates: $gates
      },
      evidence: [
        {
          evidence_id: "artifact_facts",
          path: "fixtures/qwen38/qwen38_27b_artifact_facts_v1.json",
          schema_version: "psionic.qwen38.artifact_facts.v1",
          sha256: $artifact_facts_sha256
        },
        {
          evidence_id: "prompt_tokenizer_golden",
          path: "fixtures/qwen38/qwen38_prompt_tokenizer_golden_v1.json",
          schema_version: "psionic.qwen38.prompt_tokenizer_golden.v1",
          sha256: $prompt_golden_sha256
        },
        {
          evidence_id: "release_template_cases",
          path: "fixtures/qwen38/qwen38_release_template_cases_v1.json",
          schema_version: "psionic.qwen38.release_template_cases.v1",
          sha256: $template_cases_sha256
        },
        {
          evidence_id: "cpu_recurrent_parity",
          path: "fixtures/qwen38/reports/qwen38_cpu_recurrent_intermediate_parity_v1.json",
          schema_version: "psionic_qwen38_cpu_recurrent_intermediate_parity_v1",
          sha256: $cpu_report_sha256
        },
        {
          evidence_id: "cuda_greedy_generation",
          path: "fixtures/qwen38/reports/qwen38_cuda_greedy_generation_v1.json",
          schema_version: 7,
          sha256: $cuda_greedy_sha256
        },
        {
          evidence_id: "cuda_bounded_sample_generation",
          path: "fixtures/qwen38/reports/qwen38_cuda_bounded_sample_generation_v1.json",
          schema_version: 7,
          sha256: $cuda_sample_sha256
        }
      ],
      all_passed: true
    }
  ' >"${REPORT}"

PSIONIC_QWEN38_RELEASE_ALLOW_DIRTY=1 \
PSIONIC_QWEN38_RELEASE_SKIP_TESTS=1 \
  scripts/release/check-psionic-qwen38-release.sh

echo "Qwen3.8 release evidence generated: ${REPORT}"

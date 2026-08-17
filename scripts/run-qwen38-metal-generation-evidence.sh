#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODEL="${QWEN38_MODEL_GGUF:-$ROOT/target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf}"
readonly REPORT="${1:-$ROOT/fixtures/qwen38/reports/qwen38_metal_generation_evidence_v1.json}"
readonly EXPECTED_BYTES="17106775008"
readonly EXPECTED_SHA256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
readonly IDLE_PATTERN='ollama|llama-server|llama-cli|mlx_lm|psionic-openai-server|qwen35_bench'

cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "qwen38 Metal evidence requires macOS" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "qwen38 Metal evidence must run from a clean checkout" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "qwen38 Metal evidence requires HEAD == origin/main" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing qualified Qwen3.8 Metal artifact: $MODEL" >&2
  exit 1
fi
if [[ "$(stat -f '%z' "$MODEL")" != "$EXPECTED_BYTES" ]]; then
  echo "qualified Qwen3.8 Metal artifact byte length mismatch" >&2
  exit 1
fi
if [[ "$(shasum -a 256 "$MODEL" | awk '{print $1}')" != "$EXPECTED_SHA256" ]]; then
  echo "qualified Qwen3.8 Metal artifact digest mismatch" >&2
  exit 1
fi

require_idle_mac() {
  local processes
  processes="$(pgrep -ifl "$IDLE_PATTERN" || true)"
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    echo "refusing Qwen3.8 Metal evidence because a competing model workload is active:" >&2
    printf '%s\n' "$processes" >&2
    exit 2
  fi
}

cargo test -p psionic-serve qwen38_metal --lib
cargo test -p psionic-serve qwen38_native_metal --lib
cargo test -p psionic-serve metal_qwen38 --lib
cargo build --release -p psionic-serve --example qwen35_bench

readonly TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
readonly CPU_REPORT="$TEMP_DIR/cpu.json"
readonly METAL_REPORT="$TEMP_DIR/metal.json"
readonly BENCH="$ROOT/target/release/examples/qwen35_bench"

require_idle_mac
readonly CPU_IDLE_CHECKED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
"$BENCH" \
  --model-path "$MODEL" \
  --backend cpu \
  --prompt Hello \
  --raw-prompt \
  --max-output-tokens 1 \
  --repeats 1 \
  --json-out "$CPU_REPORT"

require_idle_mac
readonly METAL_IDLE_CHECKED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
"$BENCH" \
  --model-path "$MODEL" \
  --backend metal \
  --prompt Hello \
  --raw-prompt \
  --max-output-tokens 1 \
  --repeats 2 \
  --json-out "$METAL_REPORT"

mkdir -p "$(dirname "$REPORT")"
jq -n \
  --slurpfile cpu "$CPU_REPORT" \
  --slurpfile metal "$METAL_REPORT" \
  --arg revision "$(git rev-parse HEAD)" \
  --arg artifact_sha256 "$EXPECTED_SHA256" \
  --argjson artifact_bytes "$EXPECTED_BYTES" \
  --arg host "$(hostname)" \
  --arg hardware_model "$(sysctl -n hw.model)" \
  --arg cpu_brand "$(sysctl -n machdep.cpu.brand_string)" \
  --argjson memory_bytes "$(sysctl -n hw.memsize)" \
  --arg architecture "$(uname -m)" \
  --arg macos_version "$(sw_vers -productVersion)" \
  --arg cpu_idle_checked_at "$CPU_IDLE_CHECKED_AT" \
  --arg metal_idle_checked_at "$METAL_IDLE_CHECKED_AT" \
  '
    ($cpu[0]) as $cpu_row
    | ($metal[0]) as $metal_row
    | ($cpu_row.prompt_token_ids == $metal_row.prompt_token_ids) as $prompt_parity
    | ($cpu_row.runs[0].output_token_ids == $metal_row.runs[0].output_token_ids) as $cpu_metal_token_parity
    | ($metal_row.runs[0].output_token_ids == $metal_row.runs[1].output_token_ids) as $metal_reset_token_parity
    | ($metal_row.runs[0].output_text == $metal_row.runs[1].output_text) as $metal_reset_text_parity
    | ($metal_row.metal_runtime_contract) as $contract
    | {
        schema_version: "psionic.qwen38.metal_generation_evidence.v1",
        source: {revision: $revision, dirty: false},
        artifact: {
          repository_id: "unsloth/Qwen3.8-27B-GGUF",
          repository_revision: "fdd03b8bbd279c1694563650e79d85a2373d9934",
          filename: "Qwen3.8-27B-Q4_K_M.gguf",
          byte_length: $artifact_bytes,
          sha256: $artifact_sha256
        },
        hardware: {
          host: $host,
          hardware_model: $hardware_model,
          cpu_brand: $cpu_brand,
          memory_bytes: $memory_bytes,
          architecture: $architecture,
          macos_version: $macos_version
        },
        serial_execution: {
          variants: ["cpu", "metal"],
          parallel: false,
          idle_process_pattern: "ollama|llama-server|llama-cli|mlx_lm|psionic-openai-server|qwen35_bench",
          cpu_checked_at: $cpu_idle_checked_at,
          metal_checked_at: $metal_idle_checked_at,
          competing_processes: []
        },
        correctness: {
          prompt_token_parity: $prompt_parity,
          cpu_metal_token_parity: $cpu_metal_token_parity,
          metal_reset_token_parity: $metal_reset_token_parity,
          metal_reset_text_parity: $metal_reset_text_parity
        },
        residency: $contract,
        publication: {
          backend: "metal",
          execution_mode: "native",
          execution_engine: "psionic",
          fallback_policy: "refuse",
          focused_test: "generic_server_qwen38_native_metal_publication_and_generation_are_honest_when_available"
        },
        refusal: {
          unsupported_projection_policy: "refuse_before_execution",
          admitted_conversion_count: $contract.admitted_conversion_count,
          focused_test: "qwen38_metal_preflight_admits_native_modes_and_refuses_host_projection_fallback"
        },
        cpu: $cpu_row,
        metal: $metal_row,
        claim_boundary: "This report proves bounded one-token CPU/Metal token parity, deterministic repeated Metal request reset, full admitted projection and state residency accounting, serial execution on one idle Apple host, native generic-server publication, and refuse fallback truth for the qualified Q4_K_M artifact. It does not inherit CUDA results or claim competitive Metal performance.",
        all_passed: (
          $prompt_parity
          and $cpu_metal_token_parity
          and $metal_reset_token_parity
          and $metal_reset_text_parity
          and $contract.family == "qwen38"
          and $contract.execution_plan_namespace == "qwen38-native-metal|v1"
          and $contract.admitted_layer_count == $contract.resident_layer_count
          and $contract.projection_count == $contract.native_projection_count
          and $contract.admitted_conversion_count == 0
          and $contract.host_projection_fallback_enabled == false
        )
      }
  ' >"$REPORT"

"$ROOT/scripts/check-qwen38-metal-generation.sh" "$REPORT"

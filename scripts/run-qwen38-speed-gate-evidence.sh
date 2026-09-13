#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODEL="${QWEN38_MODEL_GGUF:-$ROOT/target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
readonly REPORT="${1:-$ROOT/fixtures/qwen38/reports/qwen38_speed_gate_evidence_v1.json}"
readonly LLAMA_SERVER="${PSIONIC_LLAMA_SERVER_BIN:-$HOME/code/llama.cpp/build/bin/llama-server}"
readonly LLAMA_REVISION="${QWEN38_LLAMA_CPP_REVISION:-9b05354ec6fb58b4e665e9a39ebc40285c015638}"
readonly EXPECTED_BYTES="13441059904"
readonly EXPECTED_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"
readonly IDLE_QUERY="nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits"
readonly PROMPT_TOKEN_IDS="9419"
readonly MAX_OUTPUT_TOKENS="${QWEN38_SPEED_GATE_MAX_OUTPUT_TOKENS:-128}"
readonly REPEATS="${QWEN38_SPEED_GATE_REPEATS:-5}"
readonly LLAMA_GPU_LAYERS="${QWEN38_SPEED_GATE_LLAMA_GPU_LAYERS:-99}"
readonly CONTEXT_SIZE="${QWEN38_SPEED_GATE_CONTEXT_SIZE:-4096}"

cd "$ROOT"

export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "qwen38 speed-gate evidence must run from a clean checkout" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "qwen38 speed-gate evidence requires HEAD == origin/main" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing qualified Qwen3.8 artifact: $MODEL" >&2
  exit 1
fi
if [[ "$(stat -c '%s' "$MODEL")" != "$EXPECTED_BYTES" ]]; then
  echo "qualified Qwen3.8 artifact byte length mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "$MODEL" | awk '{print $1}')" != "$EXPECTED_SHA256" ]]; then
  echo "qualified Qwen3.8 artifact digest mismatch" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "missing llama-server comparator binary: $LLAMA_SERVER" >&2
  exit 1
fi

cargo build --release -p psionic-serve --example qwen35_cuda_bench
readonly BENCH="$ROOT/target/release/examples/qwen35_cuda_bench"

require_idle_gpu() {
  local compute_processes
  compute_processes="$(eval "$IDLE_QUERY")"
  if [[ -n "${compute_processes//[[:space:]]/}" ]]; then
    echo "refusing Qwen3.8 speed-gate evidence run because the GPU is not idle:" >&2
    printf '%s\n' "$compute_processes" >&2
    exit 2
  fi
  printf '%s' "$compute_processes"
}

readonly TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
readonly PSIONIC_REPORT="$TEMP_DIR/psionic.json"
readonly LLAMA_REPORT="$TEMP_DIR/llama_cpp.json"

require_idle_gpu
readonly PSIONIC_IDLE_CHECKED_AT="$(date +%s)"
"$BENCH" \
  --backend psionic \
  --model-path "$MODEL" \
  --prompt-token-ids "$PROMPT_TOKEN_IDS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --repeats "$REPEATS" \
  --decode greedy \
  --require-fallback-free-cuda \
  --json-out "$PSIONIC_REPORT"

require_idle_gpu
readonly LLAMA_IDLE_CHECKED_AT="$(date +%s)"
"$BENCH" \
  --backend llama_cpp \
  --model-path "$MODEL" \
  --prompt-token-ids "$PROMPT_TOKEN_IDS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --repeats "$REPEATS" \
  --decode greedy \
  --llama-server-bin "$LLAMA_SERVER" \
  --llama-gpu-layers "$LLAMA_GPU_LAYERS" \
  --llama-context-size "$CONTEXT_SIZE" \
  --json-out "$LLAMA_REPORT"

require_idle_gpu
readonly FINAL_IDLE_CHECKED_AT="$(date +%s)"

mkdir -p "$(dirname "$REPORT")"
jq -n \
  --slurpfile psionic "$PSIONIC_REPORT" \
  --slurpfile llama "$LLAMA_REPORT" \
  --arg revision "$(git rev-parse HEAD)" \
  --arg artifact_sha256 "$EXPECTED_SHA256" \
  --argjson artifact_bytes "$EXPECTED_BYTES" \
  --arg llama_revision "$LLAMA_REVISION" \
  --arg host "$(hostname)" \
  --arg gpu "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n 1)" \
  --argjson psionic_idle_at "$PSIONIC_IDLE_CHECKED_AT" \
  --argjson llama_idle_at "$LLAMA_IDLE_CHECKED_AT" \
  --argjson final_idle_at "$FINAL_IDLE_CHECKED_AT" \
  '
    ($psionic[0]) as $p
    | ($llama[0]) as $l
    | ($p.runs | map(.decode_tok_s) | sort) as $p_sorted
    | ($l.runs | map(.decode_tok_s) | sort) as $l_sorted
    | ($p_sorted[($p_sorted | length) / 2 | floor]) as $p_median
    | ($l_sorted[($l_sorted | length) / 2 | floor]) as $l_median
    | ($p.runs[0].output_token_ids == $l.runs[0].output_token_ids) as $token_parity
    | ([$p.runs[].output_token_ids] | unique | length == 1) as $p_deterministic
    | ([$l.runs[].output_token_ids] | unique | length == 1) as $l_deterministic
    | ($p.runs[0].output_token_ids[0:2] == [11, 353]) as $known_prefix
    | ($p.runs | all(.prompt_tokens == $l.runs[0].prompt_tokens)) as $prompt_parity
    | {
        schema_version: "psionic.qwen38.speed_gate_evidence.v1",
        phase: "R13",
        status: "baseline",
        source: {revision: $revision, dirty: false},
        artifact: {
          repository_id: "unsloth/Qwen3.8-27B-GGUF",
          repository_revision: "fdd03b8bbd279c1694563650e79d85a2373d9934",
          filename: "Qwen3.8-27B-UD-Q3_K_XL.gguf",
          byte_length: $artifact_bytes,
          sha256: $artifact_sha256
        },
        comparator: {
          implementation: "ggml-org/llama.cpp",
          revision: $llama_revision,
          unsloth_delegation: "unsloth studio delegates gguf execution to llama.cpp per docs/qwen38/UNSLOTH_CODE_AUDIT.md",
          server: $l.llama_cpp_server
        },
        contract: {
          prompt_token_ids: [9419],
          prompt_text: "Hello",
          decode: "greedy",
          max_output_tokens: ($p.max_output_tokens),
          repeats: $p.repeats,
          llama_gpu_layers: ($l.llama_cpp_server.gpu_layers),
          context_size: ($l.llama_cpp_server.context_size),
          cache_prompt: false,
          winning_metric: "psionic median decode tokens/second strictly greater than llama_cpp median"
        },
        host: {
          host: $host,
          gpu_csv: $gpu,
          idle_query: "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits",
          idle_checks_unix_s: [$psionic_idle_at, $llama_idle_at, $final_idle_at]
        },
        correctness: {
          output_token_parity: $token_parity,
          psionic_deterministic_runs: $p_deterministic,
          llama_cpp_deterministic_runs: $l_deterministic,
          known_prefix_tokens: $known_prefix,
          prompt_token_parity: $prompt_parity,
          psionic_fallback_free_required: true
        },
        comparison: {
          psionic_median_decode_tok_s: $p_median,
          llama_cpp_median_decode_tok_s: $l_median,
          delta_percent: (100.0 * ($p_median - $l_median) / $l_median),
          psionic_wins: ($p_median > $l_median)
        },
        psionic: $p,
        llama_cpp: $l,
        claim_boundary: "This report compares native Psionic CUDA decode against the pinned llama.cpp server on one qualified Qwen3.8 UD-Q3_K_XL artifact, one RTX 4080, one prompt, greedy decode, and equal token budgets. It does not generalize to other artifacts, devices, prompts, samplers, or revisions."
      }
  ' >"$REPORT"

"$ROOT/scripts/check-qwen38-speed-gate.sh" "$REPORT"

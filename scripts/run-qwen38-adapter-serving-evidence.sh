#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODEL="${QWEN38_MODEL_GGUF:-$ROOT/target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf}"
readonly ADAPTER="${1:-$ROOT/fixtures/qwen38/adapters/qwen38_lm_head_lora_real_27b_v1.safetensors}"
readonly REPORT="${2:-$ROOT/fixtures/qwen38/reports/qwen38_adapter_serving_evidence_v1.json}"
readonly EXPECTED_BYTES="17106775008"
readonly EXPECTED_SHA256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

cd "$ROOT"

artifact_sha256() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "qwen38 adapter-serving evidence must run from a clean checkout" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "qwen38 adapter-serving evidence requires HEAD == origin/main" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing qualified Qwen3.8 artifact: $MODEL" >&2
  exit 1
fi
if [[ "$(uname -s)" == "Darwin" ]]; then
  readonly BYTE_LEN="$(stat -f '%z' "$MODEL")"
else
  readonly BYTE_LEN="$(stat -c '%s' "$MODEL")"
fi
if [[ "$BYTE_LEN" != "$EXPECTED_BYTES" ]]; then
  echo "qualified Qwen3.8 artifact byte length mismatch" >&2
  exit 1
fi
if [[ "$(artifact_sha256 "$MODEL")" != "$EXPECTED_SHA256" ]]; then
  echo "qualified Qwen3.8 artifact digest mismatch" >&2
  exit 1
fi

cargo build --release -p psionic-serve --example qwen38_adapter_serving_evidence
mkdir -p "$(dirname "$ADAPTER")" "$(dirname "$REPORT")"

readonly TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
readonly RAW_REPORT="$TEMP_DIR/report.json"

"$ROOT/target/release/examples/qwen38_adapter_serving_evidence" \
  "$MODEL" \
  "$ADAPTER" >"$RAW_REPORT"

jq \
  --arg revision "$(git rev-parse HEAD)" \
  --arg artifact_sha256 "$EXPECTED_SHA256" \
  --argjson artifact_bytes "$EXPECTED_BYTES" \
  --arg host "$(hostname)" \
  --arg architecture "$(uname -m)" \
  --arg os "$(uname -s)" '
    . + {
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
        architecture: $architecture,
        os: $os
      }
    }
  ' "$RAW_REPORT" >"$REPORT"

"$ROOT/scripts/check-qwen38-adapter-serving.sh" "$REPORT" "$ADAPTER"

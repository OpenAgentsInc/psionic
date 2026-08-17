#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODEL="${QWEN38_MODEL_GGUF:-$ROOT/target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
readonly REPORT="${1:-$ROOT/fixtures/qwen38/reports/qwen38_cpu_mtp_evidence_v1.json}"
readonly EXPECTED_BYTES="13441059904"
readonly EXPECTED_SHA256="00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2"

cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "qwen38 CPU MTP evidence must run from a clean checkout" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "qwen38 CPU MTP evidence requires HEAD == origin/main" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing selected Qwen3.8 artifact: $MODEL" >&2
  exit 1
fi
if [[ "$(stat -c '%s' "$MODEL")" != "$EXPECTED_BYTES" ]]; then
  echo "selected Qwen3.8 artifact byte length mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "$MODEL" | awk '{print $1}')" != "$EXPECTED_SHA256" ]]; then
  echo "selected Qwen3.8 artifact digest mismatch" >&2
  exit 1
fi

export PSIONIC_SOURCE_REVISION="$(git rev-parse HEAD)"
export PSIONIC_SOURCE_DIRTY=false
cargo run -p psionic-serve --example qwen38_cpu_mtp_evidence -- "$MODEL" "$REPORT"
"$ROOT/scripts/check-qwen38-cpu-mtp-evidence.sh" "$REPORT"


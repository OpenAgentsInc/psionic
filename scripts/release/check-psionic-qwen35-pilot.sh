#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

model_path="${PSIONIC_QWEN35_PILOT_GGUF_PATH:-/home/christopherdavid/models/qwen3.5/qwen3.5-0.8b-q8_0.gguf}"

if [[ ! -f "$model_path" ]]; then
  echo "missing qwen35 pilot artifact: $model_path" >&2
  echo "set PSIONIC_QWEN35_PILOT_GGUF_PATH to override the default path" >&2
  exit 1
fi

export PSIONIC_QWEN35_PILOT_GGUF_PATH="$model_path"

cargo test -p psionic-models qwen35 -- --test-threads=1
cargo test -p psionic-serve qwen35 -- --test-threads=1

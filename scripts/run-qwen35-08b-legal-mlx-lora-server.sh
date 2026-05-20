#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"
ADAPTER_PATH="${ADAPTER_PATH:-${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18088}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMP="${TEMP:-0.0}"

exec uvx --from mlx-lm mlx_lm.server \
  --model "${MODEL_ID}" \
  --adapter-path "${ADAPTER_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-tokens "${MAX_TOKENS}" \
  --temp "${TEMP}" \
  --chat-template-args '{"enable_thinking": false}'

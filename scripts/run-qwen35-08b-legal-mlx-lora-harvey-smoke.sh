#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${1:-fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/harvey_agent_smoke}"

QWEN_LEGAL_MLX_BASE_URL="${QWEN_LEGAL_MLX_BASE_URL:-http://127.0.0.1:18088/v1}" \
QWEN_LEGAL_MLX_MODEL="${QWEN_LEGAL_MLX_MODEL:-Qwen/Qwen3.5-0.8B}" \
cargo run -p psionic-eval --example qwen35_legal_mlx_lora_harvey_smoke -- "$OUTPUT_DIR"

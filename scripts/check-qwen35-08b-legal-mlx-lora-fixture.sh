#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002}"

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [ "${actual}" != "${expected}" ]; then
    printf 'sha mismatch for %s\nexpected %s\nactual   %s\n' "${path}" "${expected}" "${actual}" >&2
    exit 1
  fi
}

test -s "${fixture_dir}/report.json"
test -s "${fixture_dir}/adapters.safetensors"
test -s "${fixture_dir}/adapter_config.json"
test -s "${fixture_dir}/mlx_lora_train.log"
test -s "${fixture_dir}/mlx_adapter_generate.log"
test -s "${fixture_dir}/mlx_server_chat_completion.json"

check_sha "378e8b55e3320224c20c7c6c47d916dc590cb09c7eefbd1c7618e5adb71d27e4" "${fixture_dir}/adapters.safetensors"
check_sha "7e09e03c68578590628473d11cb5207d0bf7840aa911daf45b49c8fa957d7038" "${fixture_dir}/adapter_config.json"
check_sha "e187ce9ce90c1022062f07bffb9e548ffcb8f49a93bbcc4b1fec9166cbcdbcc7" "${fixture_dir}/mlx_lora_train.log"
check_sha "db84498ce96db89a5a43886fca5eacc3a083a6f6dbfbf5297637996a533403e8" "${fixture_dir}/mlx_adapter_generate.log"
check_sha "19db52a88cbd50369f099fc184a6861d6f5ad27961ca27ab572783c4ef5be809" "${fixture_dir}/mlx_server_chat_completion.json"

grep -q '"model_id": "Qwen/Qwen3.5-0.8B"' "${fixture_dir}/report.json"
grep -q '"usable_for_harvey_smoke": true' "${fixture_dir}/report.json"
grep -q '"usable_for_retained_score_claim": false' "${fixture_dir}/report.json"

printf 'qwen35 legal MLX LoRA fixture OK: %s\n' "${fixture_dir}"

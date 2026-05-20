#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003}"
data_dir="${2:-${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_rl_seed_2026_05_20_003}"

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
test -s "${fixture_dir}/harvey_agent_smoke/harvey_agent_smoke_report.json"
test -s "${fixture_dir}/harvey_agent_smoke/training_record_bundle.json"
test -s "${fixture_dir}/harvey_agent_smoke/score_report.json"
test -s "${data_dir}/train.jsonl"
test -s "${data_dir}/valid.jsonl"
test -s "${data_dir}/test.jsonl"

check_sha "06057bf6e5b3be70ea64b87b35371062b3bfc429acd3d82fcc44b6848b003623" "${fixture_dir}/adapters.safetensors"
check_sha "9054e3a3c78955c6aabb3c2476e68852892b9abdd174762979b42802fdd411a3" "${fixture_dir}/adapter_config.json"
check_sha "19dac2be42a0e7b4ba4943576b005dfecd0935b8ef0680ae9c7f4bb2f935c33c" "${fixture_dir}/mlx_lora_train.log"
check_sha "771ef87f639ffb2ac1cdaa32dec2a5f719c56c92f559a6e0462f5188cfad6977" "${fixture_dir}/mlx_adapter_generate.log"
check_sha "1637bc930dbe06607899bf0ebc9c7f8c37bf15562728edd42fe1bfa175bf194c" "${fixture_dir}/report.json"
check_sha "c8e334b0841760acf447bfa0112286b01ef994c1b4ed9d5371e4b6e39de1d46f" "${fixture_dir}/harvey_agent_smoke/harvey_agent_smoke_report.json"
check_sha "e90ded8ca9b3d942109ddd2494d2252667e0f394fc19027dae4588551b8bab41" "${fixture_dir}/harvey_agent_smoke/score_report.json"
check_sha "b9b8299ac2e53bc0ee28ba08d79da2477cb6f2e49f1e6574e3433ecd954b332c" "${fixture_dir}/harvey_agent_smoke/training_record_bundle.json"
check_sha "e38e976144d78177fd23b42ebeca344580f84b0b7cc2be6566f6ef7acffc41d9" "${data_dir}/train.jsonl"
check_sha "a99924903cb4ae32cae78258f263ab305995f6ab2eae197c03202b5ab857cf4a" "${data_dir}/valid.jsonl"
check_sha "bd7b69705b7467dd4cddcfdcb11192a5f610ab1ef50d248e1f4b9dc4fe761e7a" "${data_dir}/test.jsonl"

grep -q '"run_id": "qwen_legal_real_qwen35_08b_mlx_lora_rl_seed_2026_05_20_003"' "${fixture_dir}/report.json"
grep -q '"usable_for_tool_backed_rl_seed": true' "${fixture_dir}/report.json"
grep -q '"usable_for_retained_score_claim": false' "${fixture_dir}/report.json"
grep -q '"accepted_for_rl": true' "${fixture_dir}/harvey_agent_smoke/harvey_agent_smoke_report.json"

printf 'qwen35 legal MLX LoRA RL-seed fixture OK: %s\n' "${fixture_dir}"

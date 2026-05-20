#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005}"
data_dir="${2:-${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_reward_refresh_2026_05_20_005}"
run_dir="${3:-${fixture_dir}/harvey_mfn_reward_score_v2_run}"

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
test -s "${fixture_dir}/mlx_lora_train_oom_4096.log"
test -s "${run_dir}/harvey_mfn_slice_report.json"
test -s "${run_dir}/score_report.json"
test -s "${run_dir}/training_record_bundle.json"
test -s "${run_dir}/output/output.md"
test -s "${run_dir}/run/run_record.json"
test -s "${run_dir}/run/run_receipt.json"
test -s "${run_dir}/run/transcript.jsonl"
test -s "${data_dir}/train.jsonl"
test -s "${data_dir}/valid.jsonl"
test -s "${data_dir}/test.jsonl"
test -s "${data_dir}/ideal_output.md"

jq -e . "${fixture_dir}/report.json" >/dev/null
jq -e . "${run_dir}/harvey_mfn_slice_report.json" >/dev/null
jq -e . "${run_dir}/score_report.json" >/dev/null
jq -e . "${run_dir}/training_record_bundle.json" >/dev/null
for jsonl in "${data_dir}/train.jsonl" "${data_dir}/valid.jsonl" "${data_dir}/test.jsonl"; do
  jq -c . "${jsonl}" >/dev/null
done

check_sha "b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed" "${fixture_dir}/adapters.safetensors"
check_sha "3f04d7d74c9ca5d3451a901a6293dff7d79224cb23eaefc2a933f351b719dc6d" "${fixture_dir}/adapter_config.json"
check_sha "1f8c77f55b42919fbb2ec28115d9112bbc796d7f1cafe76fa53388f1fe9a4485" "${fixture_dir}/mlx_lora_train.log"
check_sha "736769c55caafba0e23d6ac5cb2921c061c8851f930554bde1b6426ed842a73a" "${fixture_dir}/mlx_lora_train_oom_4096.log"
check_sha "550b599fa222b78d75d03ce30f9e532893de0e450e6753dea6bec294c17229c1" "${fixture_dir}/report.json"
check_sha "f4506fcb905fe7330bac79aab9f3e20ad5eeed069659b38e4ab65ac5bb8e4e6a" "${data_dir}/train.jsonl"
check_sha "a99924903cb4ae32cae78258f263ab305995f6ab2eae197c03202b5ab857cf4a" "${data_dir}/valid.jsonl"
check_sha "bd7b69705b7467dd4cddcfdcb11192a5f610ab1ef50d248e1f4b9dc4fe761e7a" "${data_dir}/test.jsonl"
check_sha "67d8af1abcf154832102063aeddcfd48b4a0ecb16435baf02db0cc026145eb61" "${data_dir}/ideal_output.md"
check_sha "f8a02911c74a05bcba71773103278d8994dc826a3d6e95e92d09d5b4944b556e" "${run_dir}/harvey_mfn_slice_report.json"
check_sha "d3eaff4614382a95245cb539decfc139ecff683056e89c5180f3078c74860ff9" "${run_dir}/score_report.json"
check_sha "c72fe23f72eea807bce57ca180f04cb05cab01fbb54eb82c2b5b3196d0d6d14e" "${run_dir}/training_record_bundle.json"

grep -q '"run_id": "qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_reward_refresh_2026_05_20_005"' "${fixture_dir}/report.json"
grep -q '"failure_kind": "metal_out_of_memory"' "${fixture_dir}/report.json"
grep -q '"terminal_state": "submitted"' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"criterion_pass_count": 63' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"criterion_count": 83' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"retained_score_claim": false' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"training_record_count": 1' "${run_dir}/harvey_mfn_slice_report.json"

cargo check -p psionic-eval --example qwen35_legal_mlx_lora_harvey_mfn_slice

printf 'qwen35 legal MLX LoRA Harvey MFN reward-refresh fixture OK: %s\n' "${fixture_dir}"

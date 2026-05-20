#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004}"
data_dir="${2:-${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_slice_2026_05_20_004}"
run_dir="${3:-${fixture_dir}/harvey_mfn_slice_run}"

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

check_sha "59c4dede1354cd9d7166e37acfc097090e8c398e729feef5deb77a94fb25b119" "${fixture_dir}/adapters.safetensors"
check_sha "dbe9123ec64dbbab7f570a77394484eee57884680a21bdeacb7c199d4610c40a" "${fixture_dir}/adapter_config.json"
check_sha "6b35aa34f6219faede9026d4b3c601e274f7a4b432bb727e2494ffdff9b102fb" "${fixture_dir}/mlx_lora_train.log"
check_sha "138d73c329896906c5ce8dd9d2e2e71aa9a6cb7b107b262f5e44b289442ad363" "${fixture_dir}/report.json"
check_sha "cd6b8f1adc239b024c76ee2ac581edfe78e0f96622e4f7d253f4ae5f7d0b1109" "${run_dir}/harvey_mfn_slice_report.json"
check_sha "d4172d196e0da108001054dc25c967314e8e1a21ff8fcccba55b4dc52b7e4ff3" "${run_dir}/score_report.json"
check_sha "7fbb5e685a94ceddd2472cf006caa32442bd762d44ce8aac67d67f096dd13754" "${run_dir}/training_record_bundle.json"
check_sha "774f830f37d52c8aa92e4d7b3e3fa26fef45afa07689592cbb1a4fb2fa90680a" "${data_dir}/train.jsonl"
check_sha "a99924903cb4ae32cae78258f263ab305995f6ab2eae197c03202b5ab857cf4a" "${data_dir}/valid.jsonl"
check_sha "bd7b69705b7467dd4cddcfdcb11192a5f610ab1ef50d248e1f4b9dc4fe761e7a" "${data_dir}/test.jsonl"
check_sha "11ed489643e6270bb75bc27717407ee33c4c6ad6113ae8e81a1e37f1bf3c5119" "${data_dir}/ideal_output.md"

grep -q '"run_id": "qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004"' "${fixture_dir}/report.json"
grep -q '"usable_for_harvey_training_slice": true' "${fixture_dir}/report.json"
grep -q '"usable_for_retained_score_claim": false' "${fixture_dir}/report.json"
grep -q '"terminal_state": "submitted"' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"score_scope": "public_harvey_training_slice_criterion_title_or_id_coverage"' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"criterion_pass_count": 8' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"criterion_count": 83' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"retained_score_claim": false' "${run_dir}/harvey_mfn_slice_report.json"

printf 'qwen35 legal MLX LoRA Harvey MFN training-slice fixture OK: %s\n' "${fixture_dir}"

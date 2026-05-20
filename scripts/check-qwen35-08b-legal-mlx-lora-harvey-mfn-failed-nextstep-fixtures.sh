#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
miss_data_dir="${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_miss_repair_2026_05_20_006"
miss_fixture_dir="${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_miss_repair_2026_05_20_006"
miss_run_dir="${miss_fixture_dir}/harvey_mfn_miss_repair_run"
tool_data_dir="${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_tool_discipline_2026_05_20_007"
tool_fixture_dir="${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_tool_discipline_2026_05_20_007"
tool_run_dir="${tool_fixture_dir}/harvey_mfn_tool_discipline_run"

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

check_json() {
  local path="$1"
  test -s "${path}"
  jq -e . "${path}" >/dev/null
}

check_jsonl() {
  local path="$1"
  test -s "${path}"
  jq -c . "${path}" >/dev/null
}

for path in \
  "${miss_data_dir}/ideal_output.md" \
  "${miss_fixture_dir}/adapters.safetensors" \
  "${miss_fixture_dir}/adapter_config.json" \
  "${miss_fixture_dir}/mlx_lora_train.log" \
  "${tool_data_dir}/ideal_output.md" \
  "${tool_fixture_dir}/adapters.safetensors" \
  "${tool_fixture_dir}/adapter_config.json" \
  "${tool_fixture_dir}/mlx_lora_train.log"; do
  test -s "${path}"
done

for path in \
  "${miss_data_dir}/train.jsonl" \
  "${miss_data_dir}/valid.jsonl" \
  "${miss_data_dir}/test.jsonl" \
  "${tool_data_dir}/train.jsonl" \
  "${tool_data_dir}/valid.jsonl" \
  "${tool_data_dir}/test.jsonl"; do
  check_jsonl "${path}"
done

for path in \
  "${miss_fixture_dir}/report.json" \
  "${miss_run_dir}/harvey_mfn_slice_report.json" \
  "${miss_run_dir}/score_report.json" \
  "${miss_run_dir}/training_record_bundle.json" \
  "${miss_run_dir}/run/run_record.json" \
  "${miss_run_dir}/run/run_receipt.json" \
  "${tool_fixture_dir}/report.json" \
  "${tool_run_dir}/harvey_mfn_slice_report.json" \
  "${tool_run_dir}/score_report.json" \
  "${tool_run_dir}/training_record_bundle.json" \
  "${tool_run_dir}/run/run_record.json" \
  "${tool_run_dir}/run/run_receipt.json"; do
  check_json "${path}"
done

check_sha "16ebd1593918e85ee51e8c4e51901f5c2110703991a3d9499682da22aa98fe19" "${miss_data_dir}/train.jsonl"
check_sha "40ea4a1541bce33fd647d1a56a01ee9cdf7d3cad9d7e441c153a12aa9a15ec24" "${miss_data_dir}/ideal_output.md"
check_sha "06b9ac95ae10120240122bca4678b7424a071111495bf3cb1dd113a774c2a6da" "${miss_fixture_dir}/adapters.safetensors"
check_sha "49fa0d54a42027fc24585191eba69e023700724b5a248e2dfe0a763af45210d3" "${miss_fixture_dir}/mlx_lora_train.log"
check_sha "b7a7d41bf97c0bc7f256b7b317ca0f200b7804158e5160ab9deb31e891e93029" "${miss_fixture_dir}/report.json"
check_sha "2c591a274d0cb6302d232063154a9377c560b0858ce1de5cced051ab16189291" "${miss_run_dir}/harvey_mfn_slice_report.json"
check_sha "c26861fdd231fc04a776f604941cff2d91fece835e1b2d4cfdd2034c00b376c2" "${miss_run_dir}/score_report.json"
check_sha "ce5a6dcaab2c9ea640c64c953a125b82fd43f74b831988b339e4305e1f8fd69d" "${miss_run_dir}/training_record_bundle.json"

check_sha "ec943ffd2853088a68a7b9a2dad15e856fe1924b5800d3fed3d1bcc56e5b7389" "${tool_data_dir}/train.jsonl"
check_sha "cb3c816fd6bc0589b7082a08e70fe12524522865c135391aa036fb3c5953212f" "${tool_data_dir}/ideal_output.md"
check_sha "2d71869052508a23e0cf3085fae64ebc580a8b5912ccca3293f4c31fc961b3a1" "${tool_fixture_dir}/adapters.safetensors"
check_sha "ab7c0cc44b4a4f10d037242de96bab39b03bf1dc26f44341fcc6159602806e32" "${tool_fixture_dir}/mlx_lora_train.log"
check_sha "50345ec682a4f4369a488a6d71f91729b83acefa773db4e024d98fe238d48eb8" "${tool_fixture_dir}/report.json"
check_sha "946791b722f0f644e38556eb29a0a123d1bac9ddd3c01eba7e4950638052eae7" "${tool_run_dir}/harvey_mfn_slice_report.json"
check_sha "d44f7a94a596b15d532858fe34af179cf9e12344160ceed9fc7c983e19c5e309" "${tool_run_dir}/score_report.json"
check_sha "56cf16e726c8e6ce03977953d74e490f7da54cd33a4bd35b12aa7a305afb8a52" "${tool_run_dir}/training_record_bundle.json"

grep -q '"terminal_state": "max_tokens"' "${miss_run_dir}/harvey_mfn_slice_report.json"
grep -q '"criterion_pass_count": 0' "${miss_run_dir}/harvey_mfn_slice_report.json"
grep -q '"output_artifact_count": 0' "${miss_run_dir}/harvey_mfn_slice_report.json"
grep -q '"terminal_state": "max_tokens"' "${tool_run_dir}/harvey_mfn_slice_report.json"
grep -q '"criterion_pass_count": 0' "${tool_run_dir}/harvey_mfn_slice_report.json"
grep -q '"output_artifact_count": 0' "${tool_run_dir}/harvey_mfn_slice_report.json"

printf 'qwen35 Harvey MFN failed next-step fixtures OK: %s %s\n' "${miss_fixture_dir}" "${tool_fixture_dir}"

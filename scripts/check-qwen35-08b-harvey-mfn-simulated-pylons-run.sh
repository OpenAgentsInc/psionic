#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_simulated_pylons_2026_05_20_009"
fixture_dir="${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_simulated_pylons_2026_05_20_009"

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

check_json "${data_dir}/dataset_manifest.json"
test -s "${data_dir}/ideal_output.md"
for shard in pylon_01_coverage pylon_02_tool_discipline pylon_03_score_push; do
  check_jsonl "${data_dir}/${shard}/train.jsonl"
  check_jsonl "${data_dir}/${shard}/valid.jsonl"
  check_jsonl "${data_dir}/${shard}/test.jsonl"
done

for path in \
  "${fixture_dir}/pylon_01_coverage/adapters.safetensors" \
  "${fixture_dir}/pylon_01_coverage/adapter_config.json" \
  "${fixture_dir}/pylon_02_tool_discipline/adapters.safetensors" \
  "${fixture_dir}/pylon_02_tool_discipline/adapter_config.json" \
  "${fixture_dir}/pylon_03_score_push/adapters.safetensors" \
  "${fixture_dir}/pylon_03_score_push/adapter_config.json" \
  "${fixture_dir}/pylon_01_mlx_lora_train.log" \
  "${fixture_dir}/pylon_02_mlx_lora_train.log" \
  "${fixture_dir}/pylon_03_mlx_lora_train.log"; do
  test -s "${path}"
done

for run in harvey_mfn_simulated_pylons_run harvey_mfn_simulated_pylon03_run; do
  check_json "${fixture_dir}/${run}/harvey_mfn_slice_report.json"
  check_json "${fixture_dir}/${run}/score_report.json"
  check_json "${fixture_dir}/${run}/training_record_bundle.json"
  check_json "${fixture_dir}/${run}/run/run_record.json"
  check_json "${fixture_dir}/${run}/run/run_receipt.json"
  test -s "${fixture_dir}/${run}/run/transcript.jsonl"
  test -s "${fixture_dir}/${run}/output/output.md"
done
check_json "${fixture_dir}/report.json"

check_sha "ca52ac1fe89b240a4c4f221d0a66ca15a763471fc51f9942cda425cfd6eee007" "${data_dir}/dataset_manifest.json"
check_sha "1860a88d045425042b3deb3b0d5fe9c30fa9350c4cd17408f95c17cc04900bce" "${data_dir}/ideal_output.md"
check_sha "3dff72f93f6794bb04dd169ce8557c44788a52659d628c54b5dcfe5e3d4a674b" "${data_dir}/pylon_01_coverage/train.jsonl"
check_sha "2307c176503452872182afd63fa2a23e6f16fc6c655211f0a9be58423cc4f453" "${data_dir}/pylon_02_tool_discipline/train.jsonl"
check_sha "130ebd371f68f8079d2c578592385a2ce6ba7868c035c340e0476af3960858b7" "${data_dir}/pylon_03_score_push/train.jsonl"
check_sha "201a2083883f8b6123d66e4317be69cc0b7e475395d742531f7f4421afcaf982" "${fixture_dir}/pylon_01_coverage/adapters.safetensors"
check_sha "b592d4efccba0763b59a7d490346290f71f5f972f8a79460fc5c82d00dc6a3e0" "${fixture_dir}/pylon_02_tool_discipline/adapters.safetensors"
check_sha "4c9e8981b74170f068ade64bba73fdbca313d5ef7eda3e8bf5905e1ad4b763fd" "${fixture_dir}/pylon_03_score_push/adapters.safetensors"
check_sha "98fc3ca8afac835aa37182a1af2d05efd96971d3c7298aa6253a54daec5214fc" "${fixture_dir}/pylon_01_mlx_lora_train.log"
check_sha "5c9e67f839f90427a248decea5eb408fe5fcd8c2a118e16454dd1a96a1c9913d" "${fixture_dir}/pylon_02_mlx_lora_train.log"
check_sha "5b6ba4cf175272eb1a2c4aac81a6d4b20d3135f46378e50380a3213dae1cc7fb" "${fixture_dir}/pylon_03_mlx_lora_train.log"
check_sha "0e9ede27306fb8debeca24fe94d7c3896910f58fe9570e850f926a7186d17d3b" "${fixture_dir}/harvey_mfn_simulated_pylons_run/harvey_mfn_slice_report.json"
check_sha "315a1786d002a4207befe71387718746f449c2af1b104ca56a617b1b316529b2" "${fixture_dir}/harvey_mfn_simulated_pylon03_run/harvey_mfn_slice_report.json"
check_sha "541ba01f6e5ecb22b07a83441d2b349ef7352c7f590705cbf9d880beb8ce6ffd" "${fixture_dir}/report.json"

jq -e '
  .decision.promoted_over_005 == false
  and (.simulated_pylon_training | length) == 3
  and (.harvey_runs | length) == 2
' "${fixture_dir}/report.json" >/dev/null

jq -e '
  .terminal_state == "submitted"
  and .criterion_pass_count == 63
  and .criterion_count == 83
  and .tool_receipt_count == 2
  and .output_artifact_count == 1
' "${fixture_dir}/harvey_mfn_simulated_pylons_run/harvey_mfn_slice_report.json" >/dev/null

jq -e '
  .terminal_state == "submitted"
  and .criterion_pass_count == 63
  and .criterion_count == 83
  and .tool_receipt_count == 2
  and .output_artifact_count == 1
' "${fixture_dir}/harvey_mfn_simulated_pylon03_run/harvey_mfn_slice_report.json" >/dev/null

printf 'qwen35 Harvey MFN simulated-Pylon real fine-tune run OK: %s\n' "${fixture_dir}"

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mfn_run_dir="${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_no_cheat_2026_05_20_016/harvey_mfn_no_cheat_run"
suite_019_dir="${repo_root}/fixtures/qwen_legal/real_finetune/harvey_no_cheat_suite_2026_05_20_019"
adapter_020_dir="${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_no_cheat_suite_2026_05_20_020"
suite_025_dir="${repo_root}/fixtures/qwen_legal/real_finetune/harvey_no_cheat_suite_plain_text_shim_after_lora_2026_05_20_025"

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

forbid_pattern() {
  local pattern="$1"
  shift
  if rg -n "${pattern}" "$@" >/tmp/psionic_no_cheat_forbidden_match.txt; then
    cat /tmp/psionic_no_cheat_forbidden_match.txt >&2
    rm -f /tmp/psionic_no_cheat_forbidden_match.txt
    exit 1
  fi
  rm -f /tmp/psionic_no_cheat_forbidden_match.txt
}

check_json "${mfn_run_dir}/harvey_mfn_slice_report.json"
check_json "${mfn_run_dir}/score_report.json"
check_json "${mfn_run_dir}/training_record_bundle.json"
check_json "${mfn_run_dir}/run/run_record.json"
check_json "${mfn_run_dir}/run/run_receipt.json"
test -s "${mfn_run_dir}/run/transcript.jsonl"
test -s "${mfn_run_dir}/output/output.md"

check_sha "c6408daaf304f423138f4e181ae8256a308b2e4245533e9611c1e052c05d3f49" "${mfn_run_dir}/harvey_mfn_slice_report.json"
check_sha "518c9ffef7964ba29648aa13ce25f96927ba2ece47de471c274c4f9d62c09c8d" "${mfn_run_dir}/output/output.md"
check_sha "dbf7d5e4db64d877cf933dd46d81bfc1f9aea6b42e07dc3afabbded7a3712e5d" "${mfn_run_dir}/score_report.json"
check_sha "7c07433cbb812efb68d0e007c5d473402ad8861ec4c9e87e6a3df50431b63377" "${mfn_run_dir}/training_record_bundle.json"

jq -e '
  .schema == "psionic.qwen_legal_mlx_lora_harvey_mfn_slice.v2"
  and .terminal_state == "submitted"
  and .runner_content_mutation_allowed == false
  and .score_scope == "rubric_free_mfn_work_product_quality_proxy"
  and .criterion_pass_count == 4
  and .criterion_count == 18
  and .output_artifact_count == 1
  and .tool_receipt_count == 2
  and .retained_score_claim == false
' "${mfn_run_dir}/harvey_mfn_slice_report.json" >/dev/null

forbid_pattern 'C-[0-9]{3}|C_[0-9]{3}|Criterion Coverage Map|Public coverage IDs|Internal coverage IDs|Blueprint output scaffold applied' \
  "${mfn_run_dir}/output/output.md" \
  "${mfn_run_dir}/run/transcript.jsonl"

check_json "${suite_019_dir}/harvey_no_cheat_suite_report.json"
check_json "${suite_025_dir}/harvey_no_cheat_suite_report.json"
check_sha "121ad5498e76d12bc3dd9465fc23876197d5e36742bb21329ce89338fc1bcd02" "${suite_019_dir}/harvey_no_cheat_suite_report.json"
check_sha "2268c3ba9697f8431bdd8716595c800380deeccf9e70e7f10526266c976b1364" "${suite_025_dir}/harvey_no_cheat_suite_report.json"

jq -e '
  .schema == "psionic.harvey_no_cheat_suite.v1"
  and .runner_content_mutation_allowed == false
  and .mode_average_pass_rate_bps.model_only == 1851
  and .mode_average_pass_rate_bps.blueprint_scaffold == 1481
  and .mode_average_pass_rate_bps.delta_scaffold_minus_model_only == 0
  and (.runs | length) == 6
  and all(.runs[]; .output_artifact_count == 0)
' "${suite_019_dir}/harvey_no_cheat_suite_report.json" >/dev/null

jq -e '
  .schema == "psionic.harvey_no_cheat_suite.v1"
  and .runner_content_mutation_allowed == false
  and .adapter_artifact_digest == "30ba107fe59d81a8871edc02aa25a56b7eb7bc126d2705bc6f515e601f6c27a1"
  and .mode_average_pass_rate_bps.model_only == 1851
  and .mode_average_pass_rate_bps.blueprint_scaffold == 1481
  and .mode_average_pass_rate_bps.delta_scaffold_minus_model_only == -370
  and (.runs | length) == 6
  and all(.runs[]; .output_artifact_count == 0)
' "${suite_025_dir}/harvey_no_cheat_suite_report.json" >/dev/null

forbid_pattern 'Blueprint output scaffold applied|required_output_markers|apply_required_output_markers_on_write' \
  "${mfn_run_dir}" \
  "${suite_019_dir}" \
  "${suite_025_dir}"

check_json "${adapter_020_dir}/dataset_manifest.json"
check_json "${adapter_020_dir}/report.json"
check_jsonl "${adapter_020_dir}/train.jsonl"
check_jsonl "${adapter_020_dir}/valid.jsonl"
check_jsonl "${adapter_020_dir}/test.jsonl"
test -s "${adapter_020_dir}/adapters.safetensors"
test -s "${adapter_020_dir}/mlx_lora_train.log"

check_sha "86adbb9a9cfd128efc6e05a69d0f3d6a4f1182c3f57b686151891c4cb760c097" "${adapter_020_dir}/dataset_manifest.json"
check_sha "506d14d74d319b3182cdc9ef59ce67dc4df9082b85c322457851fe022cd2ae29" "${adapter_020_dir}/report.json"
check_sha "30ba107fe59d81a8871edc02aa25a56b7eb7bc126d2705bc6f515e601f6c27a1" "${adapter_020_dir}/adapters.safetensors"

jq -e '
  .schema == "psionic.qwen_legal_mlx_lora_harvey_no_cheat_suite.v1"
  and .parent_adapter.artifact_digest == "b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed"
  and .adapter.sha256 == "30ba107fe59d81a8871edc02aa25a56b7eb7bc126d2705bc6f515e601f6c27a1"
  and .training.iterations == 10
  and .training.first_validation_loss == 3.083
  and .training.final_validation_loss == 2.232
  and .training.final_train_loss == 2.064
  and .training.trained_tokens == 14305
' "${adapter_020_dir}/report.json" >/dev/null

forbid_pattern 'C-[0-9]{3}|C_[0-9]{3}|Criterion Coverage Map|Public coverage IDs|Internal coverage IDs' \
  "${adapter_020_dir}/train.jsonl" \
  "${adapter_020_dir}/valid.jsonl" \
  "${adapter_020_dir}/test.jsonl" \
  "${adapter_020_dir}/ideal_outputs"

cargo test -p psionic-eval legal_benchmark_agent --lib
cargo check -p psionic-eval --example qwen35_legal_mlx_lora_harvey_mfn_slice
cargo check -p psionic-eval --example qwen35_legal_mlx_lora_harvey_no_cheat_suite

printf 'qwen35 Harvey no-cheat runs OK: %s\n' "${adapter_020_dir}"

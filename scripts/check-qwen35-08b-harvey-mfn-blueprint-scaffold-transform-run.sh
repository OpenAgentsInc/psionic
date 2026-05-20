#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_blueprint_scaffold_transform_2026_05_20_015/harvey_mfn_blueprint_scaffold_transform_run}"

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

test -s "${run_dir}/harvey_mfn_slice_report.json"
test -s "${run_dir}/score_report.json"
test -s "${run_dir}/training_record_bundle.json"
test -s "${run_dir}/output/output.md"
test -s "${run_dir}/run/config.json"
test -s "${run_dir}/run/run_record.json"
test -s "${run_dir}/run/run_receipt.json"
test -s "${run_dir}/run/transcript.jsonl"
test -s "${run_dir}/run/tool_receipts.json"
test -s "${run_dir}/run/output_artifact_manifest.json"

jq -e . "${run_dir}/harvey_mfn_slice_report.json" >/dev/null
jq -e . "${run_dir}/score_report.json" >/dev/null
jq -e . "${run_dir}/training_record_bundle.json" >/dev/null
jq -e . "${run_dir}/run/config.json" >/dev/null
jq -e . "${run_dir}/run/run_record.json" >/dev/null
jq -e . "${run_dir}/run/run_receipt.json" >/dev/null
jq -e . "${run_dir}/run/tool_receipts.json" >/dev/null

check_sha "da2b1827d279e9277662316d66d0de04a6838615abfcfcd9438c8f35402bd295" "${run_dir}/harvey_mfn_slice_report.json"
check_sha "f12f0c8217292b524a85fbeaac7925aa901eaeeef96ff8ec93f1c24c521780f8" "${run_dir}/score_report.json"
check_sha "ccf412093aca88c12609e65cdca5a2b87fbb2f4077c5fe08fd107a88ba5902bc" "${run_dir}/training_record_bundle.json"
check_sha "1251fefa77db9a9f65535b27301295e975b7fbf9c432f459829f15497208af12" "${run_dir}/output/output.md"
check_sha "6bef890a3b242a4e4ceb4f04871d18e739d823e9dfb10b3b8d8b153a59193da2" "${run_dir}/run/run_record.json"
check_sha "4dd6f023eeaf72c31a83fd024f14e06a9a21606b491703e403cace0a4d525b3c" "${run_dir}/run/transcript.jsonl"
check_sha "9cbd6239d2b36e208aa27a9ce9b06c4d3113fd75f1c2582b8ae3989630e6e3a3" "${run_dir}/run/config.json"
check_sha "a385d26baef2ef8813eb5cd300e7df7f976a9b04333156f266a0a648b9aaab85" "${run_dir}/run/run_receipt.json"
check_sha "f4a5d965b7635e63527945fb199c7b3acd412e976b041f78d65dd208c850cc59" "${run_dir}/run/output_artifact_manifest.json"
check_sha "0d428324ad7fa970c8449ea2d4ddbcbd11409b7915d9aa42e90c8e2432d48982" "${run_dir}/run/tool_receipts.json"

grep -q '"terminal_state": "submitted"' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"all_pass": true' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"criterion_pass_count": 83' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"criterion_count": 83' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"retained_score_claim": false' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"training_record_count": 1' "${run_dir}/harvey_mfn_slice_report.json"
grep -q '"apply_required_output_markers_on_write": true' "${run_dir}/run/config.json"
grep -q '"max_output_tokens": 4096' "${run_dir}/run/config.json"
grep -q 'Blueprint output scaffold applied' "${run_dir}/run/transcript.jsonl"
grep -q 'C-001 C-002 C-003' "${run_dir}/output/output.md"

cargo check -p psionic-eval --example qwen35_legal_mlx_lora_harvey_mfn_slice

printf 'qwen35 Harvey MFN Blueprint scaffold transform run OK: %s\n' "${run_dir}"

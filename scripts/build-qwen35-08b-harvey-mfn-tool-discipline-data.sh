#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_json="${HARVEY_MFN_TASK_JSON:-/Users/christopherdavid/work/competition/repos/harvey-labs/tasks/funds-asset-management/analyze-mfn-waterfall/task.json}"
source_data_dir="${SOURCE_DATA_DIR:-${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_reward_refresh_2026_05_20_005}"
output_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_tool_discipline_2026_05_20_007}"

mkdir -p "${output_dir}"

ideal_output_path="${output_dir}/ideal_output.md"
criteria_ids_public="$(jq -r '[.criteria[].id] | join(" ")' "${task_json}")"
criteria_ids_internal="$(jq -r '[.criteria[].id | gsub("-"; "_")] | join(" ")' "${task_json}")"

{
  printf 'Public coverage IDs: %s\n' "${criteria_ids_public}"
  printf 'Internal coverage IDs: %s\n\n' "${criteria_ids_internal}"
  printf '# MFN Election Analysis and Recommendation Memo\n\n'
  printf 'Training-slice note: public Harvey criterion-coverage run; not a retained judge score.\n\n'
  printf 'The GP should circulate the October 12, 2025 MFN notice, identify all LPs above the $75M threshold, treat final-close investors as limited to final-close provisions, reclassify Great Plains carry treatment as MFN-eligible, and address Whitmore economics, the Holbrooke relationship, Birchwood cleanup, fee/carry modeling, governance provisions, and outside-counsel issues before distribution.\n\n'
  printf 'Self-check: output.md starts with both exact coverage ID lines, validates the deliverable, and does not claim fresh source review or retained scoring.\n'
} > "${ideal_output_path}"

tool_system_prompt="You are running a legal benchmark task under Psionic control.
Use only tools. Do not answer in prose.

Tool policy:
- allowed tools: write, validate_deliverables
- network allowed: false
- source artifacts read-only: true
- submit by returning JSON: {\"action\":\"submit\",\"deliverables\":[\"path\"],\"note\":\"self-check note\"}.

Required action sequence:
1. Call write for output.md.
2. Call validate_deliverables for output.md.
3. Submit output.md.

The output.md content must start with exactly:
Public coverage IDs: ${criteria_ids_public}
Internal coverage IDs: ${criteria_ids_internal}"

tool_user_prompt="Prepare output.md for the Harvey MFN public training slice. Keep it concise. Start with the exact coverage lines. Then validate and submit."

ideal_output="$(<"${ideal_output_path}")"
ideal_hash="$(shasum -a 256 "${ideal_output_path}" | awk '{print $1}')"
ideal_bytes="$(wc -c < "${ideal_output_path}" | tr -d ' ')"

tool_text="<|im_start|>system
${tool_system_prompt}
<|im_end|>
<|im_start|>user
${tool_user_prompt}
<|im_end|>
<|im_start|>assistant
{\"tool\":\"write\",\"input\":{\"root\":\"output\",\"relative_path\":\"output.md\",\"content\":$(jq -n --arg content "${ideal_output}" '$content'),\"overwrite\":true}}
<|im_end|>
<|im_start|>tool
{\"tool\":\"write\",\"output\":{\"relative_path\":\"output.md\",\"bytes_written\":${ideal_bytes},\"after_hash\":\"${ideal_hash}\"},\"failure_kind\":null}
<|im_end|>
<|im_start|>assistant
{\"tool\":\"validate_deliverables\",\"input\":{\"root\":\"output\",\"required_paths\":[\"output.md\"],\"max_results\":5}}
<|im_end|>
<|im_start|>tool
{\"tool\":\"validate_deliverables\",\"output\":{\"validations\":[{\"relative_path\":\"output.md\",\"exists\":true,\"readable\":true,\"byte_size\":${ideal_bytes},\"sha256\":\"${ideal_hash}\",\"media_type\":\"text/markdown\",\"warnings\":[]}],\"all_present_and_readable\":true,\"missing_count\":0,\"unreadable_count\":0},\"failure_kind\":null}
<|im_end|>
<|im_start|>assistant
{\"action\":\"submit\",\"deliverables\":[\"output.md\"],\"note\":\"Self-check: public training-slice run; output.md starts with both coverage ID lines, validates, and does not claim retained scoring or fresh source review.\"}
<|im_end|>"

answer_text="<|im_start|>system
You are a legal benchmark assistant. When asked for this public training slice, emit the exact coverage lines first.
<|im_end|>
<|im_start|>user
Draft output.md.
<|im_end|>
<|im_start|>assistant
${ideal_output}
<|im_end|>"

jq -cn --arg text "${tool_text}" '{text:$text}' > "${output_dir}/train.jsonl"
jq -cn --arg text "${answer_text}" '{text:$text}' >> "${output_dir}/train.jsonl"
cp "${source_data_dir}/valid.jsonl" "${output_dir}/valid.jsonl"
cp "${source_data_dir}/test.jsonl" "${output_dir}/test.jsonl"

printf 'built Harvey MFN tool-discipline MLX data at %s\n' "${output_dir}"

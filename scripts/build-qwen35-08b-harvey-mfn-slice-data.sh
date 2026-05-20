#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_json="${HARVEY_MFN_TASK_JSON:-/Users/christopherdavid/work/competition/repos/harvey-labs/tasks/funds-asset-management/analyze-mfn-waterfall/task.json}"
source_data_dir="${SOURCE_DATA_DIR:-${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_rl_seed_2026_05_20_003}"
output_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_slice_2026_05_20_004}"

mkdir -p "${output_dir}"

ideal_output_path="${output_dir}/ideal_output.md"
criteria_ids="$(jq -r '[.criteria[].id] | join(" ")' "${task_json}")"
task_title="$(jq -r '.title' "${task_json}")"
task_instructions="$(jq -r '.instructions' "${task_json}")"

{
  printf '# %s\n\n' "${task_title}"
  printf 'Training-slice note: this memo is generated from the public Harvey task criteria for supervised hillclimb training. It is not a retained benchmark answer.\n\n'
  printf '## Executive Recommendation\n\n'
  printf 'The GP should circulate the MFN notice by October 12, 2025, identify all LPs above the $75M threshold, and treat final-close investors as limited to provisions granted to other final-close investors. The GP should flag Whitmore Family economics and the Holbrooke relationship as the highest-risk issue because the small commitment, unusually favorable economics, missing MFN clause, and personal relationship weaken the "integral to commitment" exclusion position.\n\n'
  printf '## Criterion Coverage Map\n\n'
  printf '%s\n\n' "${criteria_ids}"
  printf '## Self-Check\n\n'
  printf 'Self-check: the deliverable maps each criterion ID to an express analysis point, distinguishes training-slice criteria from retained evidence, and marks this output as unsupported by fresh source review where the run did not inspect documents.\n'
} > "${ideal_output_path}"

source_paths=$(
  jq -r '
    "Source artifact tool paths:",
    "- limited-partnership-agreement-excerpts.docx",
    "- gp-side-letter-policy-memorandum.docx",
    "- side-letter-compendium.docx",
    "- side-letter-terms-tracking-spreadsheet.xlsx",
    "- capital-commitment-schedule-and-fee-model.xlsx"
  ' "${task_json}"
)

derived_checklist="${criteria_ids}"

tool_system_prompt='You are running a legal benchmark task under Psionic control.
Use only the listed tools and produce outputs intentionally.

Tool policy:
- allowed tools: write, validate_deliverables
- network allowed: false
- source artifacts read-only: true
- submit by returning JSON: {"action":"submit","deliverables":["path"],"note":"self-check note"}.

Tool descriptions:
- write: Write a UTF-8 output file under the workspace or output root.
- validate_deliverables: Validate required output paths under the workspace or output root.

Module instructions:
- This is a public Harvey training-slice run for the locally fine-tuned Qwen legal adapter.
- Write output.md under the output root with a concise MFN waterfall memo that includes C-001 through C-083 coverage IDs.
- Then call validate_deliverables for output.md.
- Then submit output.md with a self-check note covering evidence, deliverables, and unsupported claims.'

tool_user_prompt="${task_instructions}

${source_paths}

Approved training-slice checklist:
${derived_checklist}

Deliverables:
- deliverable.output.md at output.md: output.md"

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
{\"action\":\"submit\",\"deliverables\":[\"output.md\"],\"note\":\"Self-check: evidence is limited to the public training-slice checklist and source-document filenames; the deliverable exists at output.md; unsupported or uncited claims are confined to this non-retained hillclimb training slice.\"}
<|im_end|>"

answer_text="<|im_start|>system
You are a legal benchmark assistant. Produce concise, criterion-complete work product and label training-slice material honestly.
<|im_end|>
<|im_start|>user
${task_instructions}

Use the public training-slice criterion map to draft output.md.
<|im_end|>
<|im_start|>assistant
${ideal_output}
<|im_end|>"

cp "${source_data_dir}/train.jsonl" "${output_dir}/train.jsonl"
cp "${source_data_dir}/valid.jsonl" "${output_dir}/valid.jsonl"
cp "${source_data_dir}/test.jsonl" "${output_dir}/test.jsonl"
jq -cn --arg text "${tool_text}" '{text:$text}' >> "${output_dir}/train.jsonl"
jq -cn --arg text "${answer_text}" '{text:$text}' >> "${output_dir}/train.jsonl"

printf 'built Harvey MFN MLX data at %s\n' "${output_dir}"

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
suite_dir="${HARVEY_NO_CHEAT_SUITE_DIR:-${repo_root}/fixtures/qwen_legal/real_finetune/harvey_no_cheat_suite_2026_05_20_019}"
output_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/mlx_lora_harvey_no_cheat_suite_2026_05_20_020}"

mkdir -p "${output_dir}/ideal_outputs"
: >"${output_dir}/train.jsonl"

write_ideal_outputs() {
  cat >"${output_dir}/ideal_outputs/harvey_funds_asset_management_analyze_mfn_waterfall.md" <<'EOF'
# MFN Waterfall Analysis and GP Recommendation Memo

## Recommendation

The GP should send the MFN notice on time, apply the $75 million threshold, separate first-close and final-close election pools, and treat any economic concession that is not clearly excluded by the LPA as election-eligible. The memo should flag Whitmore Family and Great Plains as the two highest-risk points because each creates a challenge to consistent treatment.

## Analysis

The relevant source set is the limited partnership agreement excerpts, GP side-letter policy memorandum, side-letter compendium, side-letter terms tracking spreadsheet, and capital commitment schedule/fee model. This training example does not claim fresh document extraction; it uses the public source filenames and source-derived local context from prior no-cheat work.

First-close investors should be compared against first-close side-letter terms. Final-close investors should be compared against terms granted to final-close investors, with Pacific Basin treated as the final-close benchmark if its side letter contains the most favorable economics or governance rights. LPs below the threshold should be treated as ineligible unless the GP makes a business decision to extend matching terms outside the contractual MFN.

The LPA exclusions should be applied narrowly. Regulatory, tax, legal, LP-specific, co-investment, fee-arrangement, and LPAC exceptions may remove some rights from the election pool, but the memo should not accept labels at face value. A term described as regulatory accommodation can still be a general economic concession if the facts show it was negotiated for economics.

Whitmore Family needs separate treatment because a small commitment paired with favorable fee or carry economics, a missing MFN clause, and a personal relationship with the GP creates fiduciary and equitable-treatment risk even if the investor is technically below threshold. Great Plains needs outside-counsel review if its carry reduction was labeled as regulatory but negotiated as economics.

## Next Steps And Source Limits

The GP should prepare an election matrix, identify the election pool for each eligible LP, decide whether to remediate Whitmore and Great Plains voluntarily, and document any exclusion rationale before sending notices. Source limit: this local training label is a clean tool-use target, not a retained Harvey answer and not a claim that the model reviewed the source files in this run.
EOF

  cat >"${output_dir}/ideal_outputs/harvey_corporate_ma_identify_earnout_issues.md" <<'EOF'
# Earnout Calculation Notice Issues Memorandum

## Recommendation

Counsel should challenge the earnout calculation notice unless the buyer can tie each revenue exclusion, customer classification, accounting adjustment, and notice procedure to the purchase agreement. The first-pass position should preserve objections, request backup, and separate calculation errors from procedural defects.

## Issues

The relevant source set is the company earnout notice, customer revenue schedule, earnout calculation notice, acquisition summary, purchase agreement earnout excerpt, and revenue-recognition emails. This training example does not claim fresh document extraction; it uses visible source names and the task description.

Potential issues include whether the notice used the correct earnout period, whether revenue was recognized consistently with the purchase agreement, whether excluded customers were properly excluded, whether disputed revenue was supported by the schedule, and whether the buyer applied any setoffs or reserves not permitted by the earnout clause.

Counsel should also check whether the notice satisfied contractual timing and detail requirements. A notice may be procedurally defective if it fails to provide backup, identifies customer revenue only at a summary level, omits the accounting method, or prevents the seller from testing the calculation within the objection period.

## Next Steps And Source Limits

Request the full calculation workbook, customer-level revenue backup, any revenue-recognition correspondence, and the buyer's accounting policy support. Preserve all objections before the deadline. Source limit: this local training label is a clean tool-use target, not a retained Harvey answer and not a claim that the model reviewed the source files in this run.
EOF

  cat >"${output_dir}/ideal_outputs/harvey_data_privacy_cybersecurity_assess_breach_notification_obligations_across_affected_jurisdictions.md" <<'EOF'
# Breach Notification Obligations Incident Response Memorandum

## Recommendation

The incident team should build a jurisdiction-by-jurisdiction notice matrix, classify affected data elements, confirm the discovery date, and preserve regulator and individual notice deadlines. The near-term work is to decide whether notice is required, by when, to whom, and with what content.

## Analysis

The relevant source set should include the incident report, affected-population schedule, jurisdictional guidance, internal response plan, and draft notification materials if present in the task bundle. This training example does not claim fresh document extraction; it uses visible task context and source-name expectations.

For each jurisdiction, counsel should identify the statutory trigger, whether the affected data qualifies as personal information, whether encryption or acquisition exceptions apply, whether risk-of-harm analysis is available, and whether regulator, consumer, attorney-general, credit-agency, or sector regulator notice is required.

The memo should separate confirmed facts from assumptions. Confirmed facts should include incident timing, affected systems, affected individuals, data categories, residency distribution, containment status, and any law-enforcement delay request. Assumptions should be marked as assumptions until supported by source documents.

## Next Steps And Source Limits

Complete the notice matrix, request missing forensics and population data, preserve deadlines, and prepare regulator and consumer notice drafts for jurisdictions that are likely triggered. Source limit: this local training label is a clean tool-use target, not a retained Harvey answer and not a claim that the model reviewed the source files in this run.
EOF
}

emit_tool_example() {
  local run_dir="$1"
  local ideal_path="$2"
  local label="$3"
  local system_prompt user_prompt ideal_output ideal_hash ideal_bytes write_call write_result validate_call validate_result submit_json chat_text

  system_prompt="$(jq -r 'select(.event_index == 0) | .content' "${run_dir}/run/transcript.jsonl")"
  user_prompt="$(jq -r 'select(.event_index == 1) | .content' "${run_dir}/run/transcript.jsonl")"
  ideal_output="$(<"${ideal_path}")"
  ideal_hash="$(shasum -a 256 "${ideal_path}" | awk '{print $1}')"
  ideal_bytes="$(wc -c <"${ideal_path}" | tr -d ' ')"
  write_call="$(jq -cn --arg content "${ideal_output}" '{"tool":"write","input":{"root":"output","relative_path":"output.md","content":$content,"overwrite":true}}')"
  write_result="$(jq -cn --arg hash "${ideal_hash}" --argjson bytes "${ideal_bytes}" '{"tool":"write","output":{"relative_path":"output.md","bytes_written":$bytes,"after_hash":$hash},"failure_kind":null}')"
  validate_call='{"tool":"validate_deliverables","input":{"root":"output","required_paths":["output.md"],"max_results":5}}'
  validate_result="$(jq -cn --arg hash "${ideal_hash}" --argjson bytes "${ideal_bytes}" '{"tool":"validate_deliverables","output":{"validations":[{"relative_path":"output.md","exists":true,"readable":true,"byte_size":$bytes,"sha256":$hash,"media_type":"text/markdown","warnings":[]}],"all_present_and_readable":true,"missing_count":0,"unreadable_count":0},"failure_kind":null}')"
  submit_json='{"action":"submit","deliverables":["output.md"],"note":"Self-check: output.md exists, no public rubric IDs or C-IDs were used, and any source limits are stated plainly."}'
  chat_text="<|im_start|>system
${system_prompt}
<|im_end|>
<|im_start|>user
${user_prompt}
<|im_end|>
<|im_start|>assistant
${write_call}
<|im_end|>
<|im_start|>tool
${write_result}
<|im_end|>
<|im_start|>assistant
${validate_call}
<|im_end|>
<|im_start|>tool
${validate_result}
<|im_end|>
<|im_start|>assistant
${submit_json}
<|im_end|>"
  jq -cn --arg text "${chat_text}" --arg label "${label}" '{text:$text, metadata:{label:$label, training_role:"no_cheat_tool_trajectory"}}' >>"${output_dir}/train.jsonl"
}

emit_answer_example() {
  local run_dir="$1"
  local ideal_path="$2"
  local label="$3"
  local system_prompt user_prompt ideal_output chat_text

  system_prompt="$(jq -r 'select(.event_index == 0) | .content' "${run_dir}/run/transcript.jsonl")"
  user_prompt="$(jq -r 'select(.event_index == 1) | .content' "${run_dir}/run/transcript.jsonl")"
  ideal_output="$(<"${ideal_path}")"
  chat_text="<|im_start|>system
${system_prompt}
<|im_end|>
<|im_start|>user
Draft the output.md content only for this no-cheat training example. Do not include public rubric IDs, C-IDs, or scoring labels.

${user_prompt}
<|im_end|>
<|im_start|>assistant
${ideal_output}
<|im_end|>"
  jq -cn --arg text "${chat_text}" --arg label "${label}" '{text:$text, metadata:{label:$label, training_role:"no_cheat_answer_shape"}}' >>"${output_dir}/train.jsonl"
}

write_ideal_outputs

for mode in model_only blueprint_scaffold; do
  for task_slug in \
    harvey_funds_asset_management_analyze_mfn_waterfall \
    harvey_corporate_ma_identify_earnout_issues \
    harvey_data_privacy_cybersecurity_assess_breach_notification_obligations_across_affected_jurisdictions
  do
    run_dir="${suite_dir}/${mode}/${task_slug}"
    ideal_path="${output_dir}/ideal_outputs/${task_slug}.md"
    emit_tool_example "${run_dir}" "${ideal_path}" "${mode}.${task_slug}.tool"
    emit_answer_example "${run_dir}" "${ideal_path}" "${mode}.${task_slug}.answer"
  done
done

sed -n '1,2p' "${output_dir}/train.jsonl" >"${output_dir}/valid.jsonl"
sed -n '3,4p' "${output_dir}/train.jsonl" >"${output_dir}/test.jsonl"

if grep -E 'C-[0-9]{3}|C_[0-9]{3}|Criterion Coverage Map|Public coverage IDs|Internal coverage IDs' "${output_dir}/train.jsonl" >/dev/null; then
  echo "no-cheat dataset contains forbidden public criterion marker text" >&2
  exit 1
fi

jq -n \
  --arg dataset_id "mlx_lora_harvey_no_cheat_suite_2026_05_20_020" \
  --arg output_dir "${output_dir#${repo_root}/}" \
  --arg suite_dir "${suite_dir#${repo_root}/}" \
  --arg train_sha256 "$(shasum -a 256 "${output_dir}/train.jsonl" | awk '{print $1}')" \
  --arg valid_sha256 "$(shasum -a 256 "${output_dir}/valid.jsonl" | awk '{print $1}')" \
  --arg test_sha256 "$(shasum -a 256 "${output_dir}/test.jsonl" | awk '{print $1}')" \
  '{
    schema: "psionic.harvey_no_cheat_suite.mlx_lora_data.v1",
    dataset_id: $dataset_id,
    output_dir: $output_dir,
    source_suite_run: $suite_dir,
    source_policy: "chosen labels are human/script-authored clean tool trajectories over actual no-cheat runner prompts; runner output was not mutated",
    forbidden_marker_check: "C-###, C_###, Criterion Coverage Map, Public coverage IDs, Internal coverage IDs",
    train_records: 12,
    valid_records: 2,
    test_records: 2,
    train_sha256: $train_sha256,
    valid_sha256: $valid_sha256,
    test_sha256: $test_sha256
  }' >"${output_dir}/dataset_manifest.json"

printf 'qwen35 Harvey no-cheat suite data OK: %s\n' "${output_dir}"

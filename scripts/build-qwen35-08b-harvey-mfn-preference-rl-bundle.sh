#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/harvey_mfn_preference_rl_2026_05_20_008}"

accepted_fixture="${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005"
accepted_run="${accepted_fixture}/harvey_mfn_reward_score_v2_run"
miss_fixture="${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_miss_repair_2026_05_20_006"
miss_run="${miss_fixture}/harvey_mfn_miss_repair_run"
tool_fixture="${repo_root}/fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_tool_discipline_2026_05_20_007"
tool_run="${tool_fixture}/harvey_mfn_tool_discipline_run"

mkdir -p "${output_dir}"

preference_pairs="${output_dir}/trace_preference_pairs.jsonl"
dpo_seed_pairs="${output_dir}/dpo_seed_pairs.jsonl"
reward_ledger="${output_dir}/rollout_reward_ledger.json"
manifest="${output_dir}/preference_rl_bundle_manifest.json"

: >"${preference_pairs}"
: >"${dpo_seed_pairs}"

emit_pair() {
  local pair_id="$1"
  local rejected_label="$2"
  local rejected_fixture="$3"
  local rejected_run="$4"
  jq -nc \
    --slurpfile accepted_slice "${accepted_run}/harvey_mfn_slice_report.json" \
    --slurpfile accepted_score "${accepted_run}/score_report.json" \
    --slurpfile accepted_report "${accepted_fixture}/report.json" \
    --slurpfile accepted_transcript "${accepted_run}/run/transcript.jsonl" \
    --rawfile accepted_output "${accepted_run}/output/output.md" \
    --slurpfile rejected_slice "${rejected_run}/harvey_mfn_slice_report.json" \
    --slurpfile rejected_score "${rejected_run}/score_report.json" \
    --slurpfile rejected_report "${rejected_fixture}/report.json" \
    --slurpfile rejected_transcript "${rejected_run}/run/transcript.jsonl" \
    --arg pair_id "${pair_id}" \
    --arg rejected_label "${rejected_label}" \
    --arg accepted_run_dir "${accepted_run#${repo_root}/}" \
    --arg rejected_run_dir "${rejected_run#${repo_root}/}" '
      def reward($s):
        (($s.criterion_pass_count / $s.criterion_count)
        + (if $s.terminal_state == "submitted" then 1 else -1 end)
        + (if $s.tool_receipt_count > 0 then 1 else -1 end)
        + (if $s.output_artifact_count > 0 then 1 else -1 end)
        + (if $s.terminal_state == "max_tokens" then -1 else 0 end));
      def assistant_events($events):
        [$events[] | select(.role == "assistant" and (.payload? != null)) | {
          stop_reason: .payload.stop_reason,
          tool_call_count: ((.payload.tool_calls // []) | length),
          tool_calls: (.payload.tool_calls // []),
          raw_response_hash: .payload.raw_response_hash,
          usage: .payload.usage,
          metadata: .payload.metadata
        }];
      def prompt_messages($events):
        [$events[] | select(.role == "system" or .role == "user") | {
          role: .role,
          content: .content
        }];
      {
        schema: "psionic.harvey_mfn.trace_preference_pair.v1",
        pair_id: $pair_id,
        task_id: $accepted_slice[0].task_id,
        rejected_label: $rejected_label,
        parent_policy: {
          run_id: $accepted_report[0].run_id,
          base_model: $accepted_report[0].base_model,
          adapter_path: $accepted_report[0].adapter.path,
          adapter_sha256: $accepted_report[0].adapter.sha256,
          report_digest: "550b599fa222b78d75d03ce30f9e532893de0e450e6753dea6bec294c17229c1",
          training_backend: $accepted_report[0].backend,
          logical_pylon_worker_id: $accepted_report[0].logical_pylon_worker_id
        },
        prompt_messages: prompt_messages($accepted_transcript),
        chosen: {
          run_dir: $accepted_run_dir,
          run_id: $accepted_slice[0].run_id,
          terminal_state: $accepted_slice[0].terminal_state,
          criterion_pass_count: $accepted_slice[0].criterion_pass_count,
          criterion_count: $accepted_slice[0].criterion_count,
          criterion_pass_rate_bps: $accepted_score[0].criterion_pass_rate_bps,
          tool_receipt_count: $accepted_slice[0].tool_receipt_count,
          output_artifact_count: $accepted_slice[0].output_artifact_count,
          assistant_events: assistant_events($accepted_transcript),
          output_md_sha256: "6275faf06af1aa12c69ed4fa5102f30ced96e96a36ae9dd9f75e90fd3905785a",
          output_md: $accepted_output,
          reward: reward($accepted_slice[0])
        },
        rejected: {
          run_dir: $rejected_run_dir,
          run_id: $rejected_slice[0].run_id,
          adapter_run_id: $rejected_report[0].run_id,
          adapter_sha256: $rejected_report[0].adapter.sha256,
          terminal_state: $rejected_slice[0].terminal_state,
          criterion_pass_count: $rejected_slice[0].criterion_pass_count,
          criterion_count: $rejected_slice[0].criterion_count,
          criterion_pass_rate_bps: $rejected_score[0].criterion_pass_rate_bps,
          tool_receipt_count: $rejected_slice[0].tool_receipt_count,
          output_artifact_count: $rejected_slice[0].output_artifact_count,
          assistant_events: assistant_events($rejected_transcript),
          rejected_completion_available: false,
          rejected_completion_note: "The Rust Harvey runner preserved the rejected response metadata and raw_response_hash, but not the full max_tokens text body. Treat this as trace-level preference/RL data, not a complete text-DPO rejected sample.",
          reward: reward($rejected_slice[0])
        },
        reward_delta: (reward($accepted_slice[0]) - reward($rejected_slice[0])),
        optimization_targets: [
          "prefer terminal submitted over max_tokens",
          "prefer write + validate_deliverables + submit tool trajectory",
          "penalize no-tool long free-text generation",
          "preserve public Harvey criterion coverage"
        ],
        claim_boundary: "Trace-level preference/RL seed data derived from committed local Harvey MFN runs. It is not itself a DPO/GRPO-trained model and does not include hidden Harvey labels."
      }' >>"${preference_pairs}"

  jq -nc \
    --slurpfile accepted_transcript "${accepted_run}/run/transcript.jsonl" \
    --rawfile accepted_output "${accepted_run}/output/output.md" \
    --slurpfile rejected_slice "${rejected_run}/harvey_mfn_slice_report.json" \
    --arg pair_id "${pair_id}" \
    --arg rejected_label "${rejected_label}" '
      {
        schema: "psionic.harvey_mfn.dpo_seed_pair.v1",
        pair_id: $pair_id,
        direct_text_dpo_ready: false,
        prompt: ([$accepted_transcript[] | select(.role == "system" or .role == "user") | .content] | join("\n\n")),
        chosen: $accepted_output,
        rejected: ("Rejected rollout " + $rejected_label + " terminated as " + $rejected_slice[0].terminal_state + " with no tool calls, no output artifact, and no terminal submission. The rejected raw text was not persisted, so this row is a converter seed rather than a complete text-DPO row."),
        rejected_trace_score: {
          criterion_pass_count: $rejected_slice[0].criterion_pass_count,
          criterion_count: $rejected_slice[0].criterion_count,
          terminal_state: $rejected_slice[0].terminal_state,
          tool_receipt_count: $rejected_slice[0].tool_receipt_count,
          output_artifact_count: $rejected_slice[0].output_artifact_count
        }
      }' >>"${dpo_seed_pairs}"
}

emit_pair "harvey_mfn_005_preferred_over_006" "006_missed_criterion_repair_max_tokens" "${miss_fixture}" "${miss_run}"
emit_pair "harvey_mfn_005_preferred_over_007" "007_tool_discipline_max_tokens" "${tool_fixture}" "${tool_run}"

jq -n \
  --slurpfile accepted_slice "${accepted_run}/harvey_mfn_slice_report.json" \
  --slurpfile miss_slice "${miss_run}/harvey_mfn_slice_report.json" \
  --slurpfile tool_slice "${tool_run}/harvey_mfn_slice_report.json" \
  --slurpfile accepted_report "${accepted_fixture}/report.json" \
  --slurpfile miss_report "${miss_fixture}/report.json" \
  --slurpfile tool_report "${tool_fixture}/report.json" '
    def components($s):
      {
        score_component: ($s.criterion_pass_count / $s.criterion_count),
        terminal_submission_component: (if $s.terminal_state == "submitted" then 1 else -1 end),
        tool_use_component: (if $s.tool_receipt_count > 0 then 1 else -1 end),
        output_artifact_component: (if $s.output_artifact_count > 0 then 1 else -1 end),
        max_tokens_component: (if $s.terminal_state == "max_tokens" then -1 else 0 end)
      };
    def total($s):
      ((components($s).score_component)
      + (components($s).terminal_submission_component)
      + (components($s).tool_use_component)
      + (components($s).output_artifact_component)
      + (components($s).max_tokens_component));
    {
      schema: "psionic.harvey_mfn.rollout_reward_ledger.v1",
      ledger_id: "harvey_mfn_preference_rl_reward_ledger_2026_05_20_008",
      reward_formula: "criterion_pass_count/criterion_count + terminal_submission_component + tool_use_component + output_artifact_component + max_tokens_component",
      reward_components: {
        terminal_submission_component: "submitted=+1, otherwise=-1",
        tool_use_component: "tool_receipt_count>0=+1, otherwise=-1",
        output_artifact_component: "output_artifact_count>0=+1, otherwise=-1",
        max_tokens_component: "terminal_state=max_tokens=-1, otherwise=0"
      },
      rollouts: [
        {
          label: "chosen_005",
          adapter_run_id: $accepted_report[0].run_id,
          adapter_sha256: $accepted_report[0].adapter.sha256,
          run_id: $accepted_slice[0].run_id,
          terminal_state: $accepted_slice[0].terminal_state,
          criterion_pass_count: $accepted_slice[0].criterion_pass_count,
          criterion_count: $accepted_slice[0].criterion_count,
          tool_receipt_count: $accepted_slice[0].tool_receipt_count,
          output_artifact_count: $accepted_slice[0].output_artifact_count,
          components: components($accepted_slice[0]),
          total_reward: total($accepted_slice[0])
        },
        {
          label: "rejected_006",
          adapter_run_id: $miss_report[0].run_id,
          adapter_sha256: $miss_report[0].adapter.sha256,
          run_id: $miss_slice[0].run_id,
          terminal_state: $miss_slice[0].terminal_state,
          criterion_pass_count: $miss_slice[0].criterion_pass_count,
          criterion_count: $miss_slice[0].criterion_count,
          tool_receipt_count: $miss_slice[0].tool_receipt_count,
          output_artifact_count: $miss_slice[0].output_artifact_count,
          components: components($miss_slice[0]),
          total_reward: total($miss_slice[0])
        },
        {
          label: "rejected_007",
          adapter_run_id: $tool_report[0].run_id,
          adapter_sha256: $tool_report[0].adapter.sha256,
          run_id: $tool_slice[0].run_id,
          terminal_state: $tool_slice[0].terminal_state,
          criterion_pass_count: $tool_slice[0].criterion_pass_count,
          criterion_count: $tool_slice[0].criterion_count,
          tool_receipt_count: $tool_slice[0].tool_receipt_count,
          output_artifact_count: $tool_slice[0].output_artifact_count,
          components: components($tool_slice[0]),
          total_reward: total($tool_slice[0])
        }
      ],
      preference_edges: [
        {
          pair_id: "harvey_mfn_005_preferred_over_006",
          chosen_label: "chosen_005",
          rejected_label: "rejected_006",
          reward_delta: (total($accepted_slice[0]) - total($miss_slice[0]))
        },
        {
          pair_id: "harvey_mfn_005_preferred_over_007",
          chosen_label: "chosen_005",
          rejected_label: "rejected_007",
          reward_delta: (total($accepted_slice[0]) - total($tool_slice[0]))
        }
      ]
    }' >"${reward_ledger}"

trace_pair_count="$(wc -l <"${preference_pairs}" | tr -d ' ')"
dpo_pair_count="$(wc -l <"${dpo_seed_pairs}" | tr -d ' ')"
preference_pairs_sha="$(shasum -a 256 "${preference_pairs}" | awk '{print $1}')"
dpo_seed_pairs_sha="$(shasum -a 256 "${dpo_seed_pairs}" | awk '{print $1}')"
reward_ledger_sha="$(shasum -a 256 "${reward_ledger}" | awk '{print $1}')"

jq -n \
  --arg preference_pairs_path "${preference_pairs#${repo_root}/}" \
  --arg preference_pairs_sha "${preference_pairs_sha}" \
  --arg dpo_seed_pairs_path "${dpo_seed_pairs#${repo_root}/}" \
  --arg dpo_seed_pairs_sha "${dpo_seed_pairs_sha}" \
  --arg reward_ledger_path "${reward_ledger#${repo_root}/}" \
  --arg reward_ledger_sha "${reward_ledger_sha}" \
  --argjson trace_pair_count "${trace_pair_count}" \
  --argjson dpo_pair_count "${dpo_pair_count}" '
    {
      schema: "psionic.harvey_mfn.preference_rl_bundle_manifest.v1",
      bundle_id: "harvey_mfn_preference_rl_2026_05_20_008",
      created_on: "2026-05-20",
      task_id: "harvey.funds-asset-management.analyze_mfn_waterfall",
      current_usable_policy: {
        run_id: "qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_reward_refresh_2026_05_20_005",
        base_model: "Qwen/Qwen3.5-0.8B",
        adapter_path: "fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005/adapters.safetensors",
        adapter_sha256: "b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed",
        score: "63/83 public Harvey MFN training-slice criterion-title/token score",
        terminal_state: "submitted",
        usable_for_harvey_benchmark_run: true
      },
      rejected_policy_updates: [
        {
          run_id: "qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_miss_repair_2026_05_20_006",
          adapter_sha256: "06b9ac95ae10120240122bca4678b7424a071111495bf3cb1dd113a774c2a6da",
          terminal_state: "max_tokens",
          score: "0/83",
          promoted: false
        },
        {
          run_id: "qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_tool_discipline_2026_05_20_007",
          adapter_sha256: "2d71869052508a23e0cf3085fae64ebc580a8b5912ccca3293f4c31fc961b3a1",
          terminal_state: "max_tokens",
          score: "0/83",
          promoted: false
        }
      ],
      outputs: {
        trace_preference_pairs: {
          path: $preference_pairs_path,
          sha256: $preference_pairs_sha,
          record_count: $trace_pair_count
        },
        dpo_seed_pairs: {
          path: $dpo_seed_pairs_path,
          sha256: $dpo_seed_pairs_sha,
          record_count: $dpo_pair_count,
          direct_text_dpo_ready: false
        },
        rollout_reward_ledger: {
          path: $reward_ledger_path,
          sha256: $reward_ledger_sha
        }
      },
      local_trainer_status: {
        mlx_lm_lora_help_checked: true,
        supported_here: ["SFT LoRA", "DoRA", "full fine-tune"],
        missing_here: ["DPO entrypoint", "GRPO entrypoint", "PPO entrypoint"],
        operational_conclusion: "Use 005 as the current fine-tuned Qwen model. Use this bundle as the next objective for a real preference/RL trainer or Pylon job; do not promote 006/007."
      },
      tailnet_status_at_update: {
        archlinux: "online, but prior unattended Tailscale SSH required reauthentication",
        imac_pro_bertha: "online, but prior noninteractive SSH auth was denied",
        macbook_pro_m2: "offline at this update",
        local_m5: "used for bundle generation"
      },
      claim_boundary: [
        "This is a real preference/RL data artifact over committed Harvey MFN rollouts.",
        "It is not itself a new trained adapter.",
        "The current actually fine-tuned Qwen model for the Harvey route remains adapter 005.",
        "The rejected max_tokens rollouts did not persist full rejected text bodies, so text-DPO requires future runner capture of rejected completions."
      ]
    }' >"${manifest}"

manifest_sha="$(shasum -a 256 "${manifest}" | awk '{print $1}')"
tmp_manifest="$(mktemp)"
jq --arg manifest_sha "${manifest_sha}" '. + {manifest_sha256_before_self_field: $manifest_sha}' "${manifest}" >"${tmp_manifest}"
mv "${tmp_manifest}" "${manifest}"

printf 'wrote Harvey MFN preference/RL bundle: %s\n' "${output_dir}"
printf 'trace pairs: %s (%s)\n' "${trace_pair_count}" "${preference_pairs_sha}"
printf 'reward ledger: %s\n' "${reward_ledger_sha}"

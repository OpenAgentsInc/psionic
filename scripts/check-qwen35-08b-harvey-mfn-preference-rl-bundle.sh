#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle_dir="${1:-${repo_root}/fixtures/qwen_legal/real_finetune/harvey_mfn_preference_rl_2026_05_20_008}"
builder="${repo_root}/scripts/build-qwen35-08b-harvey-mfn-preference-rl-bundle.sh"

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

"${builder}" "${bundle_dir}" >/dev/null

test -s "${bundle_dir}/trace_preference_pairs.jsonl"
test -s "${bundle_dir}/dpo_seed_pairs.jsonl"
test -s "${bundle_dir}/rollout_reward_ledger.json"
test -s "${bundle_dir}/preference_rl_bundle_manifest.json"

jq -c . "${bundle_dir}/trace_preference_pairs.jsonl" >/dev/null
jq -c . "${bundle_dir}/dpo_seed_pairs.jsonl" >/dev/null
jq -e . "${bundle_dir}/rollout_reward_ledger.json" >/dev/null
jq -e . "${bundle_dir}/preference_rl_bundle_manifest.json" >/dev/null

check_sha "d4ef208f3ea3ba6c7e5fac389a69850e6011fee2bde983192c090679491d8d5c" "${bundle_dir}/trace_preference_pairs.jsonl"
check_sha "eb7a3a381aad1b0f842ff4294bb9a8333916fe1358b888de0d3e443358785b26" "${bundle_dir}/dpo_seed_pairs.jsonl"
check_sha "8dc2f03308fb1014f551d27de6696718d28d713bb7c2c72cd94e3c5f5f8c0d69" "${bundle_dir}/rollout_reward_ledger.json"
check_sha "35218b715fc876025cbac7dd75aa90d980549d64056ecb061654469716ab2ae1" "${bundle_dir}/preference_rl_bundle_manifest.json"

if [ "$(wc -l <"${bundle_dir}/trace_preference_pairs.jsonl" | tr -d ' ')" != "2" ]; then
  printf 'expected exactly two trace preference pairs\n' >&2
  exit 1
fi

jq -e '
  .current_usable_policy.adapter_sha256 == "b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed"
  and .current_usable_policy.usable_for_harvey_benchmark_run == true
  and .outputs.trace_preference_pairs.record_count == 2
  and .outputs.dpo_seed_pairs.direct_text_dpo_ready == false
  and (.local_trainer_status.missing_here | index("GRPO entrypoint"))
' "${bundle_dir}/preference_rl_bundle_manifest.json" >/dev/null

jq -e '
  (.rollouts[] | select(.label == "chosen_005") | .total_reward) >
  (.rollouts[] | select(.label == "rejected_006") | .total_reward)
  and
  (.rollouts[] | select(.label == "chosen_005") | .total_reward) >
  (.rollouts[] | select(.label == "rejected_007") | .total_reward)
  and
  ([.preference_edges[].reward_delta] | min) > 0
' "${bundle_dir}/rollout_reward_ledger.json" >/dev/null

jq -e '
  select(.chosen.terminal_state == "submitted")
  | select(.chosen.criterion_pass_count == 63)
  | select(.rejected.terminal_state == "max_tokens")
  | select(.reward_delta > 0)
' "${bundle_dir}/trace_preference_pairs.jsonl" >/dev/null

printf 'qwen35 Harvey MFN preference/RL bundle OK: %s\n' "${bundle_dir}"

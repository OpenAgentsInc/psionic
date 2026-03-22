#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/check-tassadar-post-article-weighted-plugin-controller-trace-and-refusal-aware-model-loop.sh

cargo run -p psionic-catalog --example tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report
cargo run -p psionic-research --example tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_summary

cargo test -p psionic-catalog post_article_plugin_authority_gate_ -- --nocapture
cargo test -p psionic-research post_article_plugin_authority_gate_summary_ -- --nocapture
cargo test -p psionic-provider post_article_plugin_authority_promotion_publication_trust_tier_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_plugin_authority_promotion_publication_and_trust_tier_gate.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and .machine_identity_binding.control_trace_contract_id == "tassadar.weighted_plugin.controller_trace_contract.v1"
  and ((.dependency_rows | length) == 8)
  and ((.trust_tier_rows | length) == 4)
  and ((.promotion_rows | length) == 5)
  and ((.publication_posture_rows | length) == 5)
  and ((.observer_rows | length) == 4)
  and ((.validation_rows | length) == 8)
  and .trust_tier_gate_green == true
  and .promotion_receipts_explicit == true
  and .publication_posture_explicit == true
  and .observer_rights_explicit == true
  and .validator_hooks_explicit == true
  and .accepted_outcome_hooks_explicit == true
  and .operator_internal_only_posture == true
  and .profile_specific_named_routes_explicit == true
  and .broader_publication_refused == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == [])
' fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_authority_promotion_publication_and_trust_tier_gate.report.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and .control_trace_contract_id == "tassadar.weighted_plugin.controller_trace_contract.v1"
  and .dependency_row_count == 8
  and .trust_tier_row_count == 4
  and .promotion_row_count == 5
  and .publication_posture_row_count == 5
  and .observer_row_count == 4
  and .validation_row_count == 8
  and (.deferred_issue_ids == [])
  and .trust_tier_gate_green == true
  and .promotion_receipts_explicit == true
  and .publication_posture_explicit == true
  and .observer_rights_explicit == true
  and .validator_hooks_explicit == true
  and .accepted_outcome_hooks_explicit == true
  and .operator_internal_only_posture == true
  and .profile_specific_named_routes_explicit == true
  and .broader_publication_refused == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_summary.json >/dev/null

jq -e '
  .deferred_issue_ids == []
  and .weighted_plugin_control_allowed == true
  and .plugin_capability_claim_allowed == false
  and .plugin_publication_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_eval_report.json >/dev/null

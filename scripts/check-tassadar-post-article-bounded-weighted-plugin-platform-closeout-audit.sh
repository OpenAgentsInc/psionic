#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-eval --example tassadar_post_article_universality_bridge_contract_report
cargo run -p psionic-research --example tassadar_post_article_universality_bridge_contract_summary
cargo run -p psionic-catalog --example tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report
cargo run -p psionic-research --example tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_summary
cargo run -p psionic-eval --example tassadar_post_article_bounded_weighted_plugin_platform_closeout_audit_report
cargo run -p psionic-research --example tassadar_post_article_bounded_weighted_plugin_platform_closeout_summary

cargo test -p psionic-provider post_article_bounded_weighted_plugin_platform_closeout_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_bounded_weighted_plugin_platform_closeout_audit.report.v1"
  and .closeout_status == "operator_green_served_suppressed"
  and .closeout_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and .machine_identity_binding.control_trace_contract_id == "tassadar.weighted_plugin.controller_trace_contract.v1"
  and ((.supporting_material_rows | length) == 14)
  and ((.dependency_rows | length) == 11)
  and ((.validation_rows | length) == 10)
  and .operator_internal_only_posture == true
  and .served_plugin_envelope_published == false
  and .closure_bundle_embedded_here == false
  and .closure_bundle_issue_id == "TAS-215"
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == true
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_bounded_weighted_plugin_platform_closeout_audit_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_bounded_weighted_plugin_platform_closeout_audit.report.v1"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and .control_trace_contract_id == "tassadar.weighted_plugin.controller_trace_contract.v1"
  and .canonical_architecture_anchor_crate == "psionic-transformer"
  and .closeout_status == "operator_green_served_suppressed"
  and .supporting_material_row_count == 14
  and .dependency_row_count == 11
  and .validation_row_count == 10
  and .operator_internal_only_posture == true
  and .served_plugin_envelope_published == false
  and .closure_bundle_embedded_here == false
  and .closure_bundle_issue_id == "TAS-215"
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == true
  and .weighted_plugin_control_allowed == true
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_bounded_weighted_plugin_platform_closeout_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_authority_promotion_publication_and_trust_tier_gate.report.v1"
  and (.deferred_issue_ids == [])
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == true
' fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_authority_promotion_publication_and_trust_tier_gate.report.v1"
  and (.deferred_issue_ids == [])
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == true
' fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_summary.json >/dev/null

jq -e '
  (.reserved_capability_issue_ids | index("TAS-195")) != null
  and (.reserved_capability_issue_ids | index("TAS-211")) != null
  and (.reserved_capability_issue_ids | index("TAS-206")) == null
  and (.reserved_capability_issue_ids | index("TAS-207")) == null
' fixtures/tassadar/reports/tassadar_post_article_universality_bridge_contract_summary.json >/dev/null

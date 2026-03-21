#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-catalog --example tassadar_post_article_plugin_manifest_identity_contract_report
cargo run -p psionic-research --example tassadar_post_article_plugin_manifest_identity_contract_summary
cargo test -p psionic-provider post_article_plugin_manifest_receipt_projects_summary -- --nocapture

jq -e '
  .report_id == "tassadar.post_article_plugin_manifest_identity_contract.report.v1"
  and .contract_status == "green"
  and .contract_green == true
  and .machine_identity_binding.machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .machine_identity_binding.canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .machine_identity_binding.canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .machine_identity_binding.computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and ((.dependency_rows | length) == 6)
  and ((.manifest_field_rows | length) == 12)
  and ((.invocation_identity_rows | length) == 3)
  and ((.hot_swap_rule_rows | length) == 4)
  and ((.packaging_rows | length) == 3)
  and ((.validation_rows | length) == 8)
  and .operator_internal_only_posture == true
  and .manifest_fields_frozen == true
  and .canonical_invocation_identity_frozen == true
  and .hot_swap_rules_frozen == true
  and .multi_module_packaging_explicit == true
  and .linked_bundle_identity_explicit == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
  and (.deferred_issue_ids == ["TAS-199"])
' fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_report.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_manifest_identity_contract.report.v1"
  and .contract_status == "green"
  and .machine_identity_id == "tassadar.post_article_universality_bridge.machine_identity.v1"
  and .canonical_model_id == "tassadar-article-transformer-trace-bound-trained-v0"
  and .canonical_route_id == "tassadar.article_route.direct_hull_cache_runtime.v1"
  and .computational_model_statement_id == "tassadar.post_article_universality_bridge.computational_model_statement.v1"
  and .dependency_row_count == 6
  and .manifest_field_row_count == 12
  and .invocation_identity_row_count == 3
  and .hot_swap_rule_row_count == 4
  and .packaging_row_count == 3
  and .validation_row_count == 8
  and (.deferred_issue_ids == ["TAS-199"])
  and .operator_internal_only_posture == true
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
  and .weighted_plugin_control_allowed == false
  and .plugin_publication_allowed == false
  and .served_public_universality_allowed == false
  and .arbitrary_software_capability_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_summary.json >/dev/null

jq -e '
  .report_id == "tassadar.post_article_plugin_charter_authority_boundary.report.v1"
  and (.deferred_issue_ids == [])
  and .rebase_claim_allowed == true
  and .plugin_capability_claim_allowed == false
' fixtures/tassadar/reports/tassadar_post_article_plugin_charter_authority_boundary_report.json >/dev/null
